// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

// when using precompiled headers with this file, the deserialization of XML files will crash in release mode
//#include "Core/Precompiled.h"

#include <map>
#include <vector>

#include "tinyformat/tinyformat.h"

#include "App.h"
#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Core/Scene.h"
#include "Textures/Texture.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"

using namespace Raycer;

Scene::~Scene()
{
	RAYCER_FREE(texturesPtr);
	RAYCER_FREE(materialsPtr);
	RAYCER_FREE(trianglesPtr);
	RAYCER_FREE(emissiveTrianglesPtr);
}

void Scene::initialize()
{
	Log& log = App::getLog();
	log.logInfo("Initializing the scene");

	Timer timer;

	std::vector<Texture> allTextures;
	std::vector<Material> allMaterials;
	std::vector<Triangle> allTriangles;

	allTextures.insert(allTextures.end(), textures.begin(), textures.end());
	allMaterials.insert(allMaterials.end(), materials.begin(), materials.end());
	allTriangles.insert(allTriangles.end(), triangles.begin(), triangles.end());

	// MODEL LOADING

	if (!models.empty())
	{
		ModelLoader modelLoader;

		for (ModelLoaderInfo& modelInfo : models)
		{
			ModelLoaderResult result = modelLoader.load(modelInfo);

			allTextures.insert(allTextures.end(), result.textures.begin(), result.textures.end());
			allMaterials.insert(allMaterials.end(), result.materials.begin(), result.materials.end());
			allTriangles.insert(allTriangles.end(), result.triangles.begin(), result.triangles.end());
		}
	}

	// POINTER ASSIGNMENT & INITIALIZATION

	if (allTextures.size() > 0)
	{
		texturesPtr = static_cast<Texture*>(RAYCER_MALLOC(allTextures.size() * sizeof(Texture)));

		if (texturesPtr == nullptr)
			throw std::runtime_error("Could not allocate memory for textures");

		memcpy(texturesPtr, allTextures.data(), allTextures.size() * sizeof(Texture));
	}

	if (allMaterials.size() > 0)
	{
		materialsPtr = static_cast<Material*>(RAYCER_MALLOC(allMaterials.size() * sizeof(Material)));

		if (materialsPtr == nullptr)
			throw std::runtime_error("Could not allocate memory for materials");

		memcpy(materialsPtr, allMaterials.data(), allMaterials.size() * sizeof(Material));
	}

	std::map<uint32_t, Texture*> texturesMap;
	std::map<uint32_t, Material*> materialsMap;

	for (uint32_t i = 0; i < allTextures.size(); ++i)
	{
		if (texturesPtr[i].id == 0)
			throw std::runtime_error(tfm::format("A texture must have a non-zero id"));

		if (texturesMap.count(texturesPtr[i].id))
			throw std::runtime_error(tfm::format("A duplicate texture id was found (id: %s, type: %s)", texturesPtr[i].id));

		texturesMap[texturesPtr[i].id] = &texturesPtr[i];
		texturesPtr[i].initialize();
	}

	for (uint32_t i = 0; i < allMaterials.size(); ++i)
	{
		if (materialsPtr[i].id == 0)
			throw std::runtime_error(tfm::format("A material must have a non-zero id"));

		if (materialsMap.count(materialsPtr[i].id))
			throw std::runtime_error(tfm::format("A duplicate material id was found (id: %s)", materialsPtr[i].id));

		materialsMap[materialsPtr[i].id] = &materialsPtr[i];

		if (texturesMap.count(materialsPtr[i].emittanceTextureId))
			materialsPtr[i].emittanceTexture = texturesMap[materialsPtr[i].emittanceTextureId];

		if (texturesMap.count(materialsPtr[i].reflectanceTextureId))
			materialsPtr[i].reflectanceTexture = texturesMap[materialsPtr[i].reflectanceTextureId];

		if (texturesMap.count(materialsPtr[i].normalTextureId))
			materialsPtr[i].normalTexture = texturesMap[materialsPtr[i].normalTextureId];

		if (texturesMap.count(materialsPtr[i].maskTextureId))
			materialsPtr[i].maskTexture = texturesMap[materialsPtr[i].maskTextureId];

		if (texturesMap.count(materialsPtr[i].blinnPhongMaterial.specularReflectanceTextureId))
			materialsPtr[i].blinnPhongMaterial.specularReflectanceTexture = texturesMap[materialsPtr[i].blinnPhongMaterial.specularReflectanceTextureId];

		if (texturesMap.count(materialsPtr[i].blinnPhongMaterial.glossinessTextureId))
			materialsPtr[i].blinnPhongMaterial.glossinessTexture = texturesMap[materialsPtr[i].blinnPhongMaterial.glossinessTextureId];
	}

	std::vector<Triangle> emissiveTriangles;

	for (Triangle& triangle : allTriangles)
	{
		if (materialsMap.count(triangle.materialId))
			triangle.material = materialsMap[triangle.materialId];
		else
			throw std::runtime_error(tfm::format("A triangle has a non-existent material id (%d)", triangle.materialId));
		
		triangle.initialize();

		if (triangle.material->isEmissive())
			emissiveTriangles.push_back(triangle);
	}

	if (emissiveTriangles.size() > 0)
	{
		emissiveTrianglesPtr = static_cast<Triangle*>(RAYCER_MALLOC(emissiveTriangles.size() * sizeof(Triangle)));

		if (emissiveTrianglesPtr == nullptr)
			throw std::runtime_error("Could not allocate memory for emissive triangles");

		memcpy(emissiveTrianglesPtr, emissiveTriangles.data(), emissiveTriangles.size() * sizeof(Triangle));
		emissiveTrianglesCount = uint32_t(emissiveTriangles.size());
	}

	// BVH BUILD

	bvh.build(allTriangles);

	trianglesPtr = static_cast<Triangle*>(RAYCER_MALLOC(allTriangles.size() * sizeof(Triangle)));

	if (trianglesPtr == nullptr)
		throw std::runtime_error("Could not allocate memory for triangles");

	memcpy(trianglesPtr, allTriangles.data(), allTriangles.size() * sizeof(Triangle));

	// MISC

	camera.initialize();

	log.logInfo("Scene initialization finished (time: %s)", timer.getElapsed().getString(true));
}

CUDA_CALLABLE bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh.intersect(*this, ray, intersection);
}

CUDA_CALLABLE void Scene::calculateNormalMapping(Intersection& intersection) const
{
	if (!general.normalMapping || intersection.material->normalTexture == nullptr)
		return;

	Color normalColor = intersection.material->normalTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b * 2.0f - 1.0f);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

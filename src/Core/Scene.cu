// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

// when using precompiled headers with this file, the deserialization of XML files will crash in release mode
//#include "Precompiled.h"

#include "Precompiled.h"

#include "tinyformat/tinyformat.h"

#include "App.h"
#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Core/Scene.h"
#include "Textures/Texture.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"

using namespace Raycer;

Scene::Scene() : texturesAlloc(false), materialsAlloc(false), trianglesAlloc(false), emissiveTrianglesAlloc(false)
{
}

void Scene::initialize()
{
	Log& log = App::getLog();
	log.logInfo("Initializing the scene");

	Timer timer;

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

	// INDEX ASSIGNMENT & INITIALIZATION

	std::map<uint32_t, uint32_t> texturesMap;
	std::map<uint32_t, uint32_t> materialsMap;

	for (uint32_t i = 0; i < allTextures.size(); ++i)
	{
		if (allTextures[i].id == 0)
			throw std::runtime_error(tfm::format("A texture must have a non-zero id"));

		if (texturesMap.count(allTextures[i].id))
			throw std::runtime_error(tfm::format("A duplicate texture id was found (id: %s, type: %s)", allTextures[i].id));

		texturesMap[allTextures[i].id] = i;
		allTextures[i].initialize(*this);
	}

	for (uint32_t i = 0; i < allMaterials.size(); ++i)
	{
		if (allMaterials[i].id == 0)
			throw std::runtime_error(tfm::format("A material must have a non-zero id"));

		if (materialsMap.count(allMaterials[i].id))
			throw std::runtime_error(tfm::format("A duplicate material id was found (id: %s)", allMaterials[i].id));

		materialsMap[allMaterials[i].id] = i;

		if (texturesMap.count(allMaterials[i].emittanceTextureId))
			allMaterials[i].emittanceTextureIndex = texturesMap[allMaterials[i].emittanceTextureId];

		if (texturesMap.count(allMaterials[i].reflectanceTextureId))
			allMaterials[i].reflectanceTextureIndex = texturesMap[allMaterials[i].reflectanceTextureId];

		if (texturesMap.count(allMaterials[i].normalTextureId))
			allMaterials[i].normalTextureIndex = texturesMap[allMaterials[i].normalTextureId];

		if (texturesMap.count(allMaterials[i].maskTextureId))
			allMaterials[i].maskTextureIndex = texturesMap[allMaterials[i].maskTextureId];

		if (texturesMap.count(allMaterials[i].blinnPhongMaterial.specularReflectanceTextureId))
			allMaterials[i].blinnPhongMaterial.specularReflectanceTextureIndex = texturesMap[allMaterials[i].blinnPhongMaterial.specularReflectanceTextureId];

		if (texturesMap.count(allMaterials[i].blinnPhongMaterial.glossinessTextureId))
			allMaterials[i].blinnPhongMaterial.glossinessTextureIndex = texturesMap[allMaterials[i].blinnPhongMaterial.glossinessTextureId];
	}

	for (Triangle& triangle : allTriangles)
	{
		if (materialsMap.count(triangle.materialId))
			triangle.materialIndex = materialsMap[triangle.materialId];
		else
			throw std::runtime_error(tfm::format("A triangle has a non-existent material id (%d)", triangle.materialId));
		
		triangle.initialize();

		if (allMaterials[triangle.materialIndex].isEmissive())
			emissiveTriangles.push_back(triangle);
	}

	// BVH BUILD

	bvh.build(allTriangles);

	// MEMORY ALLOC & WRITE

	if (allTextures.size() > 0)
	{
		texturesAlloc.resize(allTextures.size());
		texturesAlloc.write(allTextures.data(), allTextures.size());
	}
	
	if (allMaterials.size() > 0)
	{
		materialsAlloc.resize(allMaterials.size());
		materialsAlloc.write(allMaterials.data(), allMaterials.size());
	}
	
	if (allTriangles.size() > 0)
	{
		trianglesAlloc.resize(allTriangles.size());
		trianglesAlloc.write(allTriangles.data(), allTriangles.size());
	}
	
	if (emissiveTriangles.size() > 0)
	{
		emissiveTrianglesAlloc.resize(emissiveTriangles.size());
		emissiveTrianglesAlloc.write(emissiveTriangles.data(), emissiveTriangles.size());
	}

	emissiveTrianglesCount = uint32_t(emissiveTriangles.size());
	
	// MISC

	camera.initialize();
	imagePool.commit();

	log.logInfo("Scene initialization finished (time: %s)", timer.getElapsed().getString(true));
}

CUDA_CALLABLE bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh.intersect(*this, ray, intersection);
}

CUDA_CALLABLE void Scene::calculateNormalMapping(Intersection& intersection) const
{
	const Material& material = getMaterial(intersection.materialIndex);

	if (!general.normalMapping || material.normalTextureIndex == -1)
		return;

	const Texture& normalTexture = getTexture(material.normalTextureIndex);
	Color normalColor = normalTexture.getColor(*this, intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b * 2.0f - 1.0f);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

CUDA_CALLABLE const Texture* Scene::getTextures() const
{
	return texturesAlloc.getPtr();
}

CUDA_CALLABLE const Material* Scene::getMaterials() const
{
	return materialsAlloc.getPtr();
}

CUDA_CALLABLE const Triangle* Scene::getTriangles() const
{
	return trianglesAlloc.getPtr();
}

CUDA_CALLABLE const Triangle* Scene::getEmissiveTriangles() const
{
	return emissiveTrianglesAlloc.getPtr();
}

CUDA_CALLABLE uint32_t Scene::getEmissiveTrianglesCount() const
{
	return emissiveTrianglesCount;
}

CUDA_CALLABLE const Texture& Scene::getTexture(uint32_t index) const
{
	return texturesAlloc.getPtr()[index];
}

CUDA_CALLABLE const Material& Scene::getMaterial(uint32_t index) const
{
	return materialsAlloc.getPtr()[index];
}

CUDA_CALLABLE const Triangle& Scene::getTriangle(uint32_t index) const
{
	return trianglesAlloc.getPtr()[index];
}

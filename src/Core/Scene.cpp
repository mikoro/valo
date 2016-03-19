// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

// when using precompiled headers with this file, the deserialization of XML files will crash in release mode
//#include "Core/Precompiled.h"

#include "Core/Precompiled.h"

#include "Core/App.h"
#include "Core/Scene.h"
#include "Textures/Texture.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"

using namespace Raycer;

Scene::~Scene()
{
	if (texturesPtr != nullptr)
	{
		free(texturesPtr);
		texturesPtr = nullptr;
	}

	if (materialsPtr != nullptr)
	{
		free(materialsPtr);
		materialsPtr = nullptr;
	}

	if (trianglesPtr != nullptr)
	{
		free(trianglesPtr);
		trianglesPtr = nullptr;
	}

	if (triangles4Ptr != nullptr)
	{
		free(triangles4Ptr);
		triangles4Ptr = nullptr;
	}

	if (emissiveTrianglesPtr != nullptr)
	{
		free(emissiveTrianglesPtr);
		emissiveTrianglesPtr = nullptr;
	}
}

Scene Scene::load(const std::string& fileName)
{
	App::getLog().logInfo("Loading scene from %s", fileName);

	std::ifstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the scene file for loading");

	Scene scene;

	cereal::XMLInputArchive archive(file);
	archive(scene);

	file.close();

	return scene;
}

void Scene::save(const std::string& fileName) const
{
	App::getLog().logInfo("Saving scene to %s", fileName);

	std::ofstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the scene file for saving");
	
	// force scope
	{
		cereal::XMLOutputArchive archive(file);
		archive(cereal::make_nvp("scene", *this));
	}

	file.close();
}

void Scene::initialize()
{
	Log& log = App::getLog();
	log.logInfo("Initializing the scene");

	Timer timer;

	std::vector<Texture> allTextures;
	std::vector<Material> allMaterials;
	std::vector<Triangle> allTriangles;
	std::vector<Triangle> emissiveTriangles;

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

	texturesPtr = static_cast<Texture*>(malloc(allTextures.size() * sizeof(Texture)));
	materialsPtr = static_cast<Material*>(malloc(allMaterials.size() * sizeof(Material)));

	memcpy(texturesPtr, &allTextures[0], allTextures.size() * sizeof(Texture));
	memcpy(materialsPtr, &allMaterials[0], allMaterials.size() * sizeof(Material));

	std::map<uint64_t, Texture*> texturesMap;
	std::map<uint64_t, Material*> materialsMap;

	for (uint64_t i = 0; i < allTextures.size(); ++i)
	{
		if (texturesPtr[i].id == 0)
			throw std::runtime_error(tfm::format("A texture must have a non-zero id"));

		if (texturesMap.count(texturesPtr[i].id))
			throw std::runtime_error(tfm::format("A duplicate texture id was found (id: %s, type: %s)", texturesPtr[i].id));

		texturesMap[texturesPtr[i].id] = &texturesPtr[i];
	}

	for (uint64_t i = 0; i < allMaterials.size(); ++i)
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

	emissiveTrianglesPtr = static_cast<Triangle*>(malloc(emissiveTriangles.size() * sizeof(Triangle)));
	memcpy(emissiveTrianglesPtr, &emissiveTriangles[0], emissiveTriangles.size() * sizeof(Triangle));
	emissiveTrianglesCount = emissiveTriangles.size();
	
	// BVH BUILD

	std::vector<TriangleSOA<4>> triangles4;
	bvh.build(allTriangles, triangles4);

	trianglesPtr = static_cast<Triangle*>(malloc(allTriangles.size() * sizeof(Triangle)));
	memcpy(trianglesPtr, &allTriangles[0], allTriangles.size() * sizeof(Triangle));

	triangles4Ptr = static_cast<TriangleSOA<4>*>(malloc(triangles4.size() * sizeof(TriangleSOA<4>)));
	memcpy(triangles4Ptr, &triangles4[0], triangles4.size() * sizeof(TriangleSOA<4>));
	
	camera.initialize();

	log.logInfo("Scene initialization finished (time: %s)", timer.getElapsed().getString(true));
}

bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh.intersect(*this, ray, intersection);
}

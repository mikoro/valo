// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

// when using precompiled headers with this file, the deserialization of XML files will crash in release mode
//#include "Precompiled.h"

#include "Precompiled.h"

#include "Tracing/Scene.h"
#include "Textures/Texture.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Utils/Timer.h"

using namespace Raycer;

Scene Scene::loadFromFile(const std::string& fileName)
{
	App::getLog().logInfo("Loading scene from %s", fileName);

	std::ifstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the scene file for loading");

	Scene scene;

	if (StringUtils::endsWith(fileName, ".json"))
	{
		cereal::JSONInputArchive archive(file);
		archive(scene);
	}
	else if (StringUtils::endsWith(fileName, ".xml"))
	{
		cereal::XMLInputArchive archive(file);
		archive(scene);
	}
	else if (StringUtils::endsWith(fileName, ".bin"))
	{
		cereal::BinaryInputArchive archive(file);
		archive(scene);
	}
	else
		throw std::runtime_error("Unsupported scene file format");

	file.close();

	return scene;
}

Scene Scene::loadFromJsonString(const std::string& text)
{
	App::getLog().logInfo("Loading scene from JSON string");

	Scene scene;
	std::stringstream ss(text);
	cereal::JSONInputArchive archive(ss);
	archive(scene);

	return scene;
}

Scene Scene::loadFromXmlString(const std::string& text)
{
	App::getLog().logInfo("Loading scene from XML string");

	Scene scene;
	std::stringstream ss(text);
	cereal::XMLInputArchive archive(ss);
	archive(scene);

	return scene;
}

void Scene::saveToFile(const std::string& fileName) const
{
	App::getLog().logInfo("Saving scene to %s", fileName);

	std::ofstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the scene file for saving");

	if (StringUtils::endsWith(fileName, ".json"))
	{
		cereal::JSONOutputArchive archive(file);
		archive(cereal::make_nvp("scene", *this));
	}
	else if (StringUtils::endsWith(fileName, ".xml"))
	{
		cereal::XMLOutputArchive archive(file);
		archive(cereal::make_nvp("scene", *this));
	}
	else if (StringUtils::endsWith(fileName, ".bin"))
	{
		cereal::BinaryOutputArchive archive(file);
		archive(cereal::make_nvp("scene", *this));
	}
	else
		throw std::runtime_error("Unsupported scene file format");

	file.close();
}

std::string Scene::getJsonString() const
{
	App::getLog().logInfo("Saving the scene to JSON string");

	std::stringstream ss;

	// force desctructor invocation for flushing
	{
		cereal::JSONOutputArchive archive(ss);
		archive(cereal::make_nvp("scene", *this));
	}

	return ss.str();
}

std::string Scene::getXmlString() const
{
	App::getLog().logInfo("Saving the scene to XML string");

	std::stringstream ss;

	// force desctructor invocation for flushing
	{
		cereal::XMLOutputArchive archive(ss);
		archive(cereal::make_nvp("scene", *this));
	}

	return ss.str();
}

void Scene::loadBvhData(const std::string& fileName)
{
	App::getLog().logInfo("Loading BVH data from %s", fileName);

	std::ifstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the BVH data file for loading");

	cereal::BinaryInputArchive archive(file);
	archive(bvhData);
	
	file.close();
}

void Scene::saveBvhData(const std::string& fileName) const
{
	App::getLog().logInfo("Saving BVH data to %s", fileName);

	std::ofstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the BVH data file for saving");

	cereal::BinaryOutputArchive archive(file);
	archive(cereal::make_nvp("bvhData", bvhData));
	
	file.close();
}

void Scene::loadImagePool(const std::string& fileName)
{
	App::getLog().logInfo("Loading image pool data from %s", fileName);

	std::ifstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the image pool data file for loading");

	cereal::BinaryInputArchive archive(file);
	archive(imagePool);

	file.close();
}

void Scene::saveImagePool(const std::string& fileName) const
{
	App::getLog().logInfo("Saving image pool data to %s", fileName);

	std::ofstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the image pool data file for saving");

	cereal::BinaryOutputArchive archive(file);
	archive(cereal::make_nvp("imagePool", imagePool));

	file.close();
}

void Scene::initialize()
{
	Log& log = App::getLog();
	log.logInfo("Initializing the scene");

	Timer timer;

	// MODEL LOADING

	if (!models.empty())
	{
		ModelLoader modelLoader;

		for (ModelLoaderInfo modelInfo : models)
		{
			if (bvhInfo.loadFromFile)
				modelInfo.loadOnlyMaterials = true;

			ModelLoaderResult result = modelLoader.load(modelInfo);

			bvhData.triangles.insert(bvhData.triangles.end(), result.triangles.begin(), result.triangles.end());
			materials.defaultMaterials.insert(materials.defaultMaterials.end(), result.defaultMaterials.begin(), result.defaultMaterials.end());
			textures.imageTextures.insert(textures.imageTextures.end(), result.imageTextures.begin(), result.imageTextures.end());
		}
	}

	// LIGHT POINTERS

	for (AmbientLight& light : lights.ambientLights)
		lightsList.push_back(&light);

	for (DirectionalLight& light : lights.directionalLights)
		lightsList.push_back(&light);

	for (PointLight& light : lights.pointLights)
		lightsList.push_back(&light);

	for (AreaPointLight& light : lights.areaPointLights)
		lightsList.push_back(&light);

	// TEXTURE POINTERS

	for (ColorTexture& texture : textures.colorTextures)
		texturesList.push_back(&texture);

	for (ColorGradientTexture& texture : textures.colorGradientTextures)
		texturesList.push_back(&texture);

	for (CheckerTexture& texture : textures.checkerTextures)
		texturesList.push_back(&texture);

	for (ImageTexture& texture : textures.imageTextures)
		texturesList.push_back(&texture);

	for (PerlinNoiseTexture& texture : textures.perlinNoiseTextures)
		texturesList.push_back(&texture);

	// MATERIAL POINTERS

	for (DefaultMaterial& material : materials.defaultMaterials)
		materialsList.push_back(&material);

	// POINTER ASSIGNMENT

	std::map<uint64_t, Texture*> texturesMap;
	std::map<uint64_t, Material*> materialsMap;

	for (Texture* texture : texturesList)
	{
		if (texture->id == 0)
			throw std::runtime_error(tfm::format("A texture must have a non-zero id (type: %s)", typeid(*texture).name()));

		if (texturesMap.count(texture->id))
			throw std::runtime_error(tfm::format("A duplicate texture id was found (id: %s, type: %s)", texture->id, typeid(*texture).name()));

		texturesMap[texture->id] = texture;
	}

	for (Material* material : materialsList)
	{
		if (material->id == 0)
			throw std::runtime_error(tfm::format("A material must have a non-zero id"));

		if (materialsMap.count(material->id))
			throw std::runtime_error(tfm::format("A duplicate material id was found (id: %s)", material->id));

		materialsMap[material->id] = material;

		if (texturesMap.count(material->reflectanceMapTextureId))
			material->reflectanceMapTexture = texturesMap[material->reflectanceMapTextureId];

		if (texturesMap.count(material->emittanceMapTextureId))
			material->emittanceMapTexture = texturesMap[material->emittanceMapTextureId];

		if (texturesMap.count(material->ambientMapTextureId))
			material->ambientMapTexture = texturesMap[material->ambientMapTextureId];

		if (texturesMap.count(material->diffuseMapTextureId))
			material->diffuseMapTexture = texturesMap[material->diffuseMapTextureId];

		if (texturesMap.count(material->specularMapTextureId))
			material->specularMapTexture = texturesMap[material->specularMapTextureId];

		if (texturesMap.count(material->normalMapTextureId))
			material->normalMapTexture = texturesMap[material->normalMapTextureId];

		if (texturesMap.count(material->maskMapTextureId))
			material->maskMapTexture = texturesMap[material->maskMapTextureId];
	}

	// BVH LOADING

	if (bvhInfo.loadFromFile)
		loadBvhData(bvhInfo.fileName);

	// TRIANGLE INITIALIZATION

	for (Triangle& triangle : bvhData.triangles)
	{
		if (materialsMap.count(triangle.materialId))
			triangle.material = materialsMap[triangle.materialId];
		else
			throw std::runtime_error(tfm::format("A triangle has a non-existent material id (%d)", triangle.materialId));

		if (!bvhInfo.loadFromFile)
			triangle.initialize();
	}

	// BVH BUILDING

	switch (bvhInfo.bvhType)
	{
		case BVHType::BVH1: bvh = &bvhData.bvh1; break;
		case BVHType::BVH4: bvh = &bvhData.bvh4; break;
		default: break;
	}

	if (!bvhInfo.loadFromFile)
		bvh->build(*this);
	
	// EMISSIVE TRIANGLES

	for (Triangle& triangle : bvhData.triangles)
	{
		if (triangle.material->isEmissive())
			emissiveTriangles.push_back(&triangle);
	}

	// IMAGE POOL LOADING

	if (imagePoolInfo.loadFromFile)
		loadImagePool(imagePoolInfo.fileName);
	
	// MISC INITIALIZATION

	for (Light* light : lightsList)
		light->initialize();

	for (Texture* texture : texturesList)
		texture->initialize(*this);	

	camera.initialize();

	log.logInfo("Scene initialization finished (time: %s)", timer.getElapsed().getString(true));
}

bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh->intersect(*this, ray, intersection);
}

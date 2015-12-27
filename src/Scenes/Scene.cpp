// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

// when using precompiled headers with this file, the deserialization of XML files will crash in release mode
//#include "Precompiled.h"

#include "tinyformat/tinyformat.h"

#include "Scenes/Scene.h"
#include "Textures/Texture.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Utils/Timer.h"

#include "cereal/archives/json.hpp"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/binary.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"

using namespace Raycer;

Scene Scene::createTestScene(uint64_t number)
{
	App::getLog().logInfo("Creating test scene number %d", number);

	switch (number)
	{
		case 1: return createTestScene1();
		case 2: return createTestScene2();
		case 3: return createTestScene3();
		default: throw std::runtime_error("Unknown test scene number");
	}
}

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

void Scene::initialize()
{
	Log& log = App::getLog();
	log.logInfo("Initializing scene");

	Timer timer;

	// MODELS

	for (const ModelLoaderInfo& modelInfo : models)
	{
		ModelLoaderResult result = ModelLoader::load(modelInfo);

		for (const Triangle& triangle : result.triangles)
			triangles.push_back(triangle);

		for (const Material& material : result.materials)
			materials.push_back(material);

		for (const ImageTexture& imageTexture : result.textures)
			textures.imageTextures.push_back(imageTexture);
	}

	models.clear();

	// TEXTURE POINTERS

	std::vector<Texture*> texturesList;

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

	// POINTER MAP GENERATION

	for (Material& material : materials)
	{
		if (material.id == 0)
			throw std::runtime_error(tfm::format("A material must have a non-zero id"));

		if (materialsMap.count(material.id))
			throw std::runtime_error(tfm::format("A duplicate material id was found (id: %s)", material.id));

		materialsMap[material.id] = &material;
	}

	for (Texture* texture : texturesList)
	{
		if (texture->id == 0)
			throw std::runtime_error(tfm::format("A texture must have a non-zero id (type: %s)", typeid(*texture).name()));

		if (texturesMap.count(texture->id))
			throw std::runtime_error(tfm::format("A duplicate texture id was found (id: %s, type: %s)", texture->id, typeid(*texture).name()));

		texturesMap[texture->id] = texture;
	}

	for (Triangle& triangle : triangles)
	{
		if (triangle.id == 0)
			throw std::runtime_error(tfm::format("A triangle must have a non-zero id"));

		if (trianglesMap.count(triangle.id))
			throw std::runtime_error(tfm::format("A duplicate triangle id was found (id: %s)", triangle.id));

		trianglesMap[triangle.id] = &triangle;

		if (materialsMap.count(triangle.materialId))
			triangle.material = materialsMap[triangle.materialId];
		else
			throw std::runtime_error(tfm::format("A triangle has a non-existent material id (%d)", triangle.materialId));

		triangle.initialize();
	}

	// POINTER SETTING

	for (Material& material : materials)
	{
		if (texturesMap.count(material.ambientMapTextureId))
			material.ambientMapTexture = texturesMap[material.ambientMapTextureId];

		if (texturesMap.count(material.diffuseMapTextureId))
			material.diffuseMapTexture = texturesMap[material.diffuseMapTextureId];

		if (texturesMap.count(material.specularMapTextureId))
			material.specularMapTexture = texturesMap[material.specularMapTextureId];

		if (texturesMap.count(material.emittanceMapTextureId))
			material.emittanceMapTexture = texturesMap[material.emittanceMapTextureId];

		if (texturesMap.count(material.normalMapTextureId))
			material.normalMapTexture = texturesMap[material.normalMapTextureId];

		if (texturesMap.count(material.maskMapTextureId))
			material.maskMapTexture = texturesMap[material.maskMapTextureId];
	}

	// INITIALIZATION

	for (Texture* texture : texturesList)
		texture->initialize(*this);

	// CAMERA

	camera.initialize();

	// BVH BUILD

	if (bvh.hasBeenBuilt())
		bvh.restore(*this);
	else
		bvh.build(triangles, bvhBuildInfo);

	log.logInfo("Scene initialization finished (time: %.2f ms)", timer.getElapsedMilliseconds());
}

bool Scene::intersect(const Ray& ray, Intersection& intersection) const
{
	return bvh.intersect(ray, intersection);
}

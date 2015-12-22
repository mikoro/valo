// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/Camera.h"
#include "Tracing/Triangle.h"
#include "Tracing/BVH.h"
#include "Textures/ColorTexture.h"
#include "Textures/ColorGradientTexture.h"
#include "Textures/CheckerTexture.h"
#include "Textures/ImageTexture.h"
#include "Textures/PerlinNoiseTexture.h"
#include "Tracing/Material.h"
#include "Tracing/Lights.h"
#include "Tracers/Tracer.h"
#include "Samplers/Sampler.h"
#include "Filters/Filter.h"
#include "Tonemappers/Tonemapper.h"
#include "Rendering/Color.h"
#include "Utils/ModelLoader.h"

namespace Raycer
{
	class Primitive;

	class Scene
	{
	public:

		static Scene createTestScene(uint64_t number);
		static Scene loadFromFile(const std::string& fileName);
		static Scene loadFromJsonString(const std::string& text);
		static Scene loadFromXmlString(const std::string& text);

		void saveToFile(const std::string& fileName) const;
		std::string getJsonString() const;
		std::string getXmlString() const;

		void initialize();

		static Scene createTestScene1();

		static const uint64_t TEST_SCENE_COUNT = 1;

		struct General
		{
			TracerType tracerType = TracerType::RAY;
			uint64_t maxRayIterations = 3;
			uint64_t maxPathLength = 3;
			uint64_t pathSampleCount = 1;
			double rayStartOffset = 0.00001;
			Color backgroundColor = Color(0.0, 0.0, 0.0);
			Color offLensColor = Color(0.0, 0.0, 0.0);
			SamplerType multiSamplerType = SamplerType::CMJ;
			FilterType multiSamplerFilterType = FilterType::MITCHELL;
			uint64_t multiSampleCountSqrt = 1;
			SamplerType timeSamplerType = SamplerType::JITTERED;
			uint64_t timeSampleCount = 1;
			SamplerType cameraSamplerType = SamplerType::CMJ;
			uint64_t cameraSampleCountSqrt = 1;
			bool enableNormalMapping = true;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(tracerType),
					CEREAL_NVP(maxRayIterations),
					CEREAL_NVP(maxPathLength),
					CEREAL_NVP(pathSampleCount),
					CEREAL_NVP(rayStartOffset),
					CEREAL_NVP(backgroundColor),
					CEREAL_NVP(offLensColor),
					CEREAL_NVP(multiSamplerType),
					CEREAL_NVP(multiSamplerFilterType),
					CEREAL_NVP(multiSampleCountSqrt),
					CEREAL_NVP(timeSamplerType),
					CEREAL_NVP(timeSampleCount),
					CEREAL_NVP(cameraSamplerType),
					CEREAL_NVP(cameraSampleCountSqrt),
					CEREAL_NVP(enableNormalMapping));
			}

		} general;

		Camera camera;

		struct Tonemapper
		{
			TonemapperType type = TonemapperType::LINEAR;
			bool applyGamma = true;
			bool shouldClamp = true;
			double gamma = 2.2;
			double exposure = 0.0;
			double key = 0.18;
			bool enableAveraging = false;
			double averagingAlpha = 0.1;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(type),
					CEREAL_NVP(applyGamma),
					CEREAL_NVP(shouldClamp),
					CEREAL_NVP(gamma),
					CEREAL_NVP(exposure),
					CEREAL_NVP(key),
					CEREAL_NVP(enableAveraging),
					CEREAL_NVP(averagingAlpha));
			}

		} tonemapper;

		struct SimpleFog
		{
			bool enabled = false;
			Color color = Color::WHITE;
			double distance = 100.0;
			double steepness = 1.0;
			bool heightDispersion = false;
			double height = 100.0;
			double heightSteepness = 1.0;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(enabled),
					CEREAL_NVP(color),
					CEREAL_NVP(distance),
					CEREAL_NVP(steepness),
					CEREAL_NVP(heightDispersion),
					CEREAL_NVP(height),
					CEREAL_NVP(heightSteepness));
			}

		} simpleFog;

		struct Textures
		{
			std::vector<ColorTexture> colorTextures;
			std::vector<ColorGradientTexture> colorGradientTextures;
			std::vector<CheckerTexture> checkerTextures;
			std::vector<ImageTexture> imageTextures;
			std::vector<PerlinNoiseTexture> perlinNoiseTextures;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(colorTextures),
					CEREAL_NVP(colorGradientTextures),
					CEREAL_NVP(checkerTextures),
					CEREAL_NVP(imageTextures),
					CEREAL_NVP(perlinNoiseTextures));
			}

		} textures;

		std::vector<Material> materials;

		struct Lights
		{
			AmbientLight ambientLight;
			std::vector<DirectionalLight> directionalLights;
			std::vector<PointLight> pointLights;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(ambientLight),
					CEREAL_NVP(directionalLights),
					CEREAL_NVP(pointLights));
			}

		} lights;

		std::vector<ModelLoaderInfo> models;
		BVHBuildInfo bvhBuildInfo;
		BVH bvh;
		std::vector<Triangle> triangles;

		std::map<uint64_t, Triangle*> trianglesMap;
		std::map<uint64_t, Material*> materialsMap;
		std::map<uint64_t, Texture*> texturesMap;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(general),
				CEREAL_NVP(camera),
				CEREAL_NVP(tonemapper),
				CEREAL_NVP(simpleFog),
				CEREAL_NVP(textures),
				CEREAL_NVP(materials),
				CEREAL_NVP(lights),
				CEREAL_NVP(models),
				CEREAL_NVP(bvhBuildInfo),
				CEREAL_NVP(bvh),
				CEREAL_NVP(triangles));
		}
	};
}

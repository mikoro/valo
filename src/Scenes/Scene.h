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
		bool intersect(const Ray& ray, Intersection& intersection) const;

		static Scene createTestScene1();

		static const uint64_t TEST_SCENE_COUNT = 1;

		struct General
		{
			TracerType tracerType = TracerType::RAY;
			double rayStartOffset = 0.00001;
			Color backgroundColor = Color(0.0, 0.0, 0.0);
			Color offLensColor = Color(0.0, 0.0, 0.0);
			bool enableNormalMapping = true;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(tracerType),
					CEREAL_NVP(rayStartOffset),
					CEREAL_NVP(backgroundColor),
					CEREAL_NVP(offLensColor),
					CEREAL_NVP(enableNormalMapping));
			}

		} general;

		struct Raytracer
		{
			uint64_t maxRayIterations = 3;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(maxRayIterations));
			}

		} raytracing;

		struct Pathtracer
		{
			uint64_t maxPathLength = 3;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(maxPathLength));
			}

		} pathtracing;

		struct Sampling
		{
			uint64_t pixelSampleCount = 1;
			uint64_t multiSampleCountSqrt = 1;
			uint64_t timeSampleCount = 1;
			uint64_t cameraSampleCountSqrt = 1;
			SamplerType multiSamplerType = SamplerType::CMJ;
			FilterType multiSamplerFilterType = FilterType::MITCHELL;
			SamplerType timeSamplerType = SamplerType::JITTERED;
			SamplerType cameraSamplerType = SamplerType::CMJ;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(pixelSampleCount),
					CEREAL_NVP(multiSampleCountSqrt),
					CEREAL_NVP(timeSampleCount),
					CEREAL_NVP(cameraSampleCountSqrt),
					CEREAL_NVP(multiSamplerType),
					CEREAL_NVP(multiSamplerFilterType),
					CEREAL_NVP(timeSamplerType),
					CEREAL_NVP(cameraSamplerType));
			}

		} sampling;

		struct Tonemapper
		{
			TonemapperType type = TonemapperType::LINEAR;
			bool applyGamma = true;
			bool shouldClamp = true;
			double gamma = 2.2;
			double exposure = 0.0;
			double key = 0.18;
			bool enableAveraging = true;
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

		} tonemapping;

		Camera camera;

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
				CEREAL_NVP(raytracing),
				CEREAL_NVP(pathtracing),
				CEREAL_NVP(sampling),
				CEREAL_NVP(tonemapping),
				CEREAL_NVP(camera),
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

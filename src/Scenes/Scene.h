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
#include "Lights/AmbientLight.h"
#include "Lights/DirectionalLight.h"
#include "Lights/PointLight.h"
#include "Lights/AreaPointLight.h"
#include "Materials/DiffuseMaterial.h"
#include "Materials/DiffuseSpecularMaterial.h"
#include "Tracers/Tracer.h"
#include "Samplers/Sampler.h"
#include "Filters/Filter.h"
#include "Tonemappers/Tonemapper.h"
#include "Rendering/Color.h"
#include "Rendering/ImagePool.h"
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
		static Scene createTestScene2();
		static Scene createTestScene3();

		static const uint64_t TEST_SCENE_COUNT = 3;

		Camera camera;

		struct General
		{
			TracerType tracerType = TracerType::RAY;
			double rayMinDistance = 0.00000000001;
			Color backgroundColor = Color(0.0, 0.0, 0.0);
			Color offLensColor = Color(0.0, 0.0, 0.0);
			bool enableNormalMapping = true;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(tracerType),
					CEREAL_NVP(rayMinDistance),
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
			double terminationProbability = 0.5;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(terminationProbability));
			}

		} pathtracing;

		struct Sampling
		{
			uint64_t pixelSampleCount = 1;
			uint64_t multiSampleCountSqrt = 1;
			uint64_t cameraSampleCountSqrt = 1;
			SamplerType multiSamplerType = SamplerType::CMJ;
			FilterType multiSamplerFilterType = FilterType::MITCHELL;
			SamplerType cameraSamplerType = SamplerType::CMJ;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(pixelSampleCount),
					CEREAL_NVP(multiSampleCountSqrt),
					CEREAL_NVP(cameraSampleCountSqrt),
					CEREAL_NVP(multiSamplerType),
					CEREAL_NVP(multiSamplerFilterType),
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

		struct Lights
		{
			std::vector<AmbientLight> ambientLights;
			std::vector<DirectionalLight> directionalLights;
			std::vector<PointLight> pointLights;
			std::vector<AreaPointLight> areaPointLights;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(ambientLights),
					CEREAL_NVP(directionalLights),
					CEREAL_NVP(pointLights),
					CEREAL_NVP(areaPointLights));
			}

		} lights;

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

		struct Materials
		{
			std::vector<DiffuseMaterial> diffuseMaterials;
			std::vector<DiffuseSpecularMaterial> diffuseSpecularMaterials;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(diffuseMaterials),
					CEREAL_NVP(diffuseSpecularMaterials));
			}

		} materials;

		std::vector<ModelLoaderInfo> models;
		BVHBuildInfo bvhBuildInfo;
		BVH bvh;
		std::vector<Triangle> triangles;
		std::vector<Triangle*> emissiveTriangles;
		ImagePool imagePool;

		std::vector<Light*> lightsList;
		std::vector<Texture*> texturesList;
		std::vector<Material*> materialsList;
		
		std::map<uint64_t, Texture*> texturesMap;
		std::map<uint64_t, Material*> materialsMap;
		std::map<uint64_t, Triangle*> trianglesMap;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(camera),
				CEREAL_NVP(general),
				CEREAL_NVP(raytracing),
				CEREAL_NVP(pathtracing),
				CEREAL_NVP(sampling),
				CEREAL_NVP(tonemapping),
				CEREAL_NVP(lights),
				CEREAL_NVP(textures),
				CEREAL_NVP(materials),
				CEREAL_NVP(models),
				CEREAL_NVP(bvhBuildInfo),
				CEREAL_NVP(bvh),
				CEREAL_NVP(triangles),
				CEREAL_NVP(imagePool));
		}
	};
}

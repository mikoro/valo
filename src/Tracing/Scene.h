// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/Camera.h"
#include "Textures/ColorTexture.h"
#include "Textures/ColorGradientTexture.h"
#include "Textures/CheckerTexture.h"
#include "Textures/ImageTexture.h"
#include "Textures/PerlinNoiseTexture.h"
#include "Textures/ValueNoiseTexture.h"
#include "Textures/CellNoiseTexture.h"
#include "Textures/MarbleTexture.h"
#include "Textures/WoodTexture.h"
#include "Textures/FireTexture.h"
#include "Textures/AtmosphereTexture.h"
#include "Textures/VoronoiTexture.h"
#include "Tracing/Material.h"
#include "Tracing/Lights.h"
#include "Primitives/Plane.h"
#include "Primitives/Sphere.h"
#include "Primitives/Box.h"
#include "Primitives/Triangle.h"
#include "Primitives/Cylinder.h"
#include "Primitives/Torus.h"
#include "Primitives/Instance.h"
#include "Primitives/FlatBVH.h"
#include "Primitives/CSG.h"
#include "Primitives/BlinnBlob.h"
#include "Primitives/PrimitiveGroup.h"
#include "Tracing/Tracer.h"
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

		Scene();

		static Scene createTestScene(uint64_t number);
		static Scene loadFromFile(const std::string& fileName);
		static Scene loadFromJsonString(const std::string& text);
		static Scene loadFromXmlString(const std::string& text);

		void saveToFile(const std::string& fileName) const;
		std::string getJsonString() const;
		std::string getXmlString() const;

		void addModel(const ModelLoaderResult& result);
		void initialize();
		void rebuildRootBVH();

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
			bool visualizeDepth = false;
			double visualizeDepthMaxDistance = 25.0;
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
					CEREAL_NVP(visualizeDepth),
					CEREAL_NVP(visualizeDepthMaxDistance),
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

		struct VolumetricFog
		{
			bool enabled = false;
			Color color = Color::WHITE;
			double density = 1.0;
			uint64_t steps = 0;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(enabled),
					CEREAL_NVP(color),
					CEREAL_NVP(density),
					CEREAL_NVP(steps));
			}

		} volumetricFog;

		struct RootBVH
		{
			bool enabled = false;
			BVHBuildInfo buildInfo;
			FlatBVH bvh;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(enabled),
					CEREAL_NVP(buildInfo),
					CEREAL_NVP(bvh));
			}

		} rootBVH;

		struct BoundingBoxes
		{
			bool enabled = false;
			bool useDefaultMaterial = true;
			Material material;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(enabled),
					CEREAL_NVP(useDefaultMaterial),
					CEREAL_NVP(material));
			}

		} boundingBoxes;

		struct Textures
		{
			std::vector<ColorTexture> colorTextures;
			std::vector<ColorGradientTexture> colorGradientTextures;
			std::vector<CheckerTexture> checkerTextures;
			std::vector<ImageTexture> imageTextures;
			std::vector<PerlinNoiseTexture> perlinNoiseTextures;
			std::vector<ValueNoiseTexture> valueNoiseTextures;
			std::vector<CellNoiseTexture> cellNoiseTextures;
			std::vector<MarbleTexture> marbleTextures;
			std::vector<WoodTexture> woodTextures;
			std::vector<FireTexture> fireTextures;
			std::vector<AtmosphereTexture> atmosphereTextures;
			std::vector<VoronoiTexture> voronoiTextures;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(colorTextures),
					CEREAL_NVP(colorGradientTextures),
					CEREAL_NVP(checkerTextures),
					CEREAL_NVP(imageTextures),
					CEREAL_NVP(perlinNoiseTextures),
					CEREAL_NVP(valueNoiseTextures),
					CEREAL_NVP(cellNoiseTextures),
					CEREAL_NVP(marbleTextures),
					CEREAL_NVP(woodTextures),
					CEREAL_NVP(fireTextures),
					CEREAL_NVP(atmosphereTextures),
					CEREAL_NVP(voronoiTextures));
			}

		} textures;

		std::vector<Material> materials;
		Material defaultMaterial;

		struct Lights
		{
			AmbientLight ambientLight;
			std::vector<DirectionalLight> directionalLights;
			std::vector<PointLight> pointLights;
			std::vector<SpotLight> spotLights;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(ambientLight),
					CEREAL_NVP(directionalLights),
					CEREAL_NVP(pointLights),
					CEREAL_NVP(spotLights));
			}

		} lights;

		std::vector<ModelLoaderInfo> models;

		struct Primitives
		{
			std::vector<Triangle> triangles;
			std::vector<Plane> planes;
			std::vector<Sphere> spheres;
			std::vector<Box> boxes;
			std::vector<Cylinder> cylinders;
			std::vector<Torus> toruses;
			std::vector<BlinnBlob> blinnBlobs;
			std::vector<CSG> csgs;
			std::vector<PrimitiveGroup> primitiveGroups;
			std::vector<Instance> instances;
			std::vector<Box> boundingBoxes;
			std::vector<Primitive*> visible;
			std::vector<Primitive*> invisible;
			std::vector<Primitive*> visibleOriginal;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(triangles),
					CEREAL_NVP(planes),
					CEREAL_NVP(spheres),
					CEREAL_NVP(boxes),
					CEREAL_NVP(cylinders),
					CEREAL_NVP(toruses),
					CEREAL_NVP(blinnBlobs),
					CEREAL_NVP(csgs),
					CEREAL_NVP(primitiveGroups),
					CEREAL_NVP(instances));
			}

		} primitives;

		std::map<uint64_t, Primitive*> primitivesMap;
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
				CEREAL_NVP(volumetricFog),
				CEREAL_NVP(rootBVH),
				CEREAL_NVP(boundingBoxes),
				CEREAL_NVP(textures),
				CEREAL_NVP(materials),
				CEREAL_NVP(lights),
				CEREAL_NVP(models),
				CEREAL_NVP(primitives));
		}
	};
}

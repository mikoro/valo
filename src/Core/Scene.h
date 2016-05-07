// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>

#include "BVH/BVH.h"
#include "Core/Camera.h"
#include "Core/Common.h"
#include "Utils/CudaAlloc.h"
#include "Core/ImagePool.h"
#include "Filters/Filter.h"
#include "Integrators/Integrator.h"
#include "Materials/Material.h"
#include "Math/Color.h"
#include "Textures/Texture.h"
#include "Tonemappers/Tonemapper.h"
#include "Utils/ModelLoader.h"

namespace Raycer
{
	class Scene
	{
	public:

		Scene();

		void initialize();

		CUDA_CALLABLE bool intersect(const Ray& ray, Intersection& intersection) const;
		CUDA_CALLABLE void calculateNormalMapping(Intersection& intersection) const;

		CUDA_CALLABLE const Texture* getTextures() const;
		CUDA_CALLABLE const Material* getMaterials() const;
		CUDA_CALLABLE const Triangle* getTriangles() const;
		CUDA_CALLABLE const Triangle* getEmissiveTriangles() const;
		CUDA_CALLABLE uint32_t getEmissiveTrianglesCount() const;

		CUDA_CALLABLE const Texture& getTexture(uint32_t index) const;
		CUDA_CALLABLE const Material& getMaterial(uint32_t index) const;
		CUDA_CALLABLE const Triangle& getTriangle(uint32_t index) const;

		struct General
		{
			float rayMinDistance = 0.0001f;
			Color backgroundColor = Color(0.0f, 0.0f, 0.0f);
			Color offLensColor = Color(0.0f, 0.0f, 0.0f);
			bool normalMapping = true;
			bool normalInterpolation = true;
			bool normalVisualization = false;
			bool interpolationVisualization = false;

		} general;

		struct Renderer
		{
			bool filtering = true;
			Filter filter;

		} renderer;

		struct Volume
		{
			bool enabled = false;
			bool attenuation = false;
			bool emission = false;
			bool inscatter = false;
			Color attenuationColor = Color(1.0f, 1.0f, 1.0f);
			Color emissionColor = Color(1.0f, 1.0f, 1.0f);
			Color inscatterColor = Color(1.0f, 1.0f, 1.0f);
			float attenuationFactor = 0.0f;
			float emissionFactor = 0.0f;
			float inscatterFactor = 0.0f;
			float stepSize = 0.1f;
			bool constant = true;
			PerlinNoise noiseDensity;
			uint32_t noiseSeed = 1;
			float noiseScale = 1.0f;

		} volume;

		Camera camera;
		Integrator integrator;
		Tonemapper tonemapper;
		BVH bvh;
		ImagePool imagePool;

		std::vector<ModelLoaderInfo> models;
		std::vector<Texture> textures;
		std::vector<Material> materials;
		std::vector<Triangle> triangles;

	private:

		std::vector<Texture> allTextures;
		std::vector<Material> allMaterials;
		std::vector<Triangle> allTriangles;
		std::vector<Triangle> emissiveTriangles;

		CudaAlloc<Texture> texturesAlloc;
		CudaAlloc<Material> materialsAlloc;
		CudaAlloc<Triangle> trianglesAlloc;
		CudaAlloc<Triangle> emissiveTrianglesAlloc;
		uint32_t emissiveTrianglesCount = 0;	
	};
}

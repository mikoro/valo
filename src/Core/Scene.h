// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/BVH.h"
#include "Core/Camera.h"
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

		~Scene();

		static Scene load(const std::string& fileName);
		void save(const std::string& fileName) const;

		void initialize();
		bool intersect(const Ray& ray, Intersection& intersection) const;
		void calculateNormalMapping(Intersection& intersection) const;

		struct General
		{
			float rayMinDistance = 0.0001f;
			Color backgroundColor = Color(0.0f, 0.0f, 0.0f);
			Color offLensColor = Color(0.0f, 0.0f, 0.0f);
			bool normalMapping = true;
			bool normalInterpolation = true;
			bool normalVisualization = false;
			bool interpolationVisualization = false;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(rayMinDistance),
					CEREAL_NVP(backgroundColor),
					CEREAL_NVP(offLensColor),
					CEREAL_NVP(normalMapping),
					CEREAL_NVP(normalInterpolation),
					CEREAL_NVP(normalVisualization),
					CEREAL_NVP(interpolationVisualization));
			}

		} general;

		struct Renderer
		{
			bool filtering = true;
			Filter filter;
			uint32_t pixelSamples = 1;

			template <class Archive>
			void serialize(Archive& ar)
			{
				ar(CEREAL_NVP(filtering),
					CEREAL_NVP(filter),
					CEREAL_NVP(pixelSamples));
			}

		} renderer;

		Camera camera;
		Integrator integrator;
		Tonemapper tonemapper;
		BVH bvh;

		std::vector<ModelLoaderInfo> models;
		std::vector<Texture> textures;
		std::vector<Material> materials;
		std::vector<Triangle> triangles;

		Texture* texturesPtr = nullptr;
		Material* materialsPtr = nullptr;
		Triangle* trianglesPtr = nullptr;
		Triangle* emissiveTrianglesPtr = nullptr;
		uint32_t emissiveTrianglesCount = 0;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(general),
				CEREAL_NVP(renderer),
				CEREAL_NVP(camera),
				CEREAL_NVP(integrator),
				CEREAL_NVP(tonemapper),
				CEREAL_NVP(bvh),
				CEREAL_NVP(models),
				CEREAL_NVP(textures),
				CEREAL_NVP(materials),
				CEREAL_NVP(triangles));
		}
	};
}

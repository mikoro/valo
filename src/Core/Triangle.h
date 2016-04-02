// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

#include "Core/Common.h"
#include "Math/Vector3.h"
#include "Math/Vector2.h"

namespace Raycer
{
	class Scene;
	class Ray;
	class Intersection;
	class Random;
	class AABB;

	template <uint32_t N>
	struct TriangleSOA
	{
		float vertex1X[N];
		float vertex1Y[N];
		float vertex1Z[N];
		float vertex2X[N];
		float vertex2Y[N];
		float vertex2Z[N];
		float vertex3X[N];
		float vertex3Y[N];
		float vertex3Z[N];
		uint32_t triangleIndex[N];
	};

	class Triangle
	{
	public:

		void initialize();
		CUDA_CALLABLE bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

		template <uint32_t N>
		CUDA_CALLABLE static bool intersect(const float* __restrict vertex1X, const float* __restrict vertex1Y, const float* __restrict vertex1Z, const float* __restrict vertex2X, const float* __restrict vertex2Y, const float* __restrict vertex2Z, const float* __restrict vertex3X, const float* __restrict vertex3Y, const float* __restrict vertex3Z, const uint32_t* __restrict triangleIndices, const Scene& scene, const Ray& ray, Intersection& intersection);

		CUDA_CALLABLE Intersection getRandomIntersection(const Scene& scene, Random& random) const;
		AABB getAABB() const;

		Vector3 vertices[3];
		Vector3 normals[3];
		Vector2 texcoords[3];
		Vector3 normal;
		Vector3 tangent;
		Vector3 bitangent;
		float area = 0.0f;
		uint32_t materialId = 0;
		uint32_t materialIndex = 0;
		
	private:

		template <uint32_t N>
		CUDA_CALLABLE static bool findIntersectionValues(const uint32_t* hits, const float* distances, const float* uValues, const float* vValues, const uint32_t* triangleIndices, float& distance, float& u, float& v, uint32_t& triangleIndex);

		CUDA_CALLABLE static bool calculateIntersectionData(const Scene& scene, const Ray& ray, const Triangle& triangle, Intersection& intersection, float distance, float u, float v);
	};
}

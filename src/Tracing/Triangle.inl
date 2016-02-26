// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

namespace Raycer
{
	template <uint64_t N>
	bool Triangle::intersect(
		const float* __restrict vertex1X,
		const float* __restrict vertex1Y,
		const float* __restrict vertex1Z,
		const float* __restrict vertex2X,
		const float* __restrict vertex2Y,
		const float* __restrict vertex2Z,
		const float* __restrict vertex3X,
		const float* __restrict vertex3Y,
		const float* __restrict vertex3Z,
		const uint32_t* __restrict triangleIndices,
		const Scene& scene,
		const Ray& ray,
		Intersection& intersection)
	{
		if (ray.fastOcclusion && intersection.wasFound)
			return true;

		const float originX = ray.origin.x;
		const float originY = ray.origin.y;
		const float originZ = ray.origin.z;

		const float directionX = ray.direction.x;
		const float directionY = ray.direction.y;
		const float directionZ = ray.direction.z;

		const float minDistance = ray.minDistance;
		const float maxDistance = ray.maxDistance;
		const float intersectionDistance = intersection.distance;

		alignas(16) uint32_t hits[N];
		alignas(16) float distances[N];
		alignas(16) float uValues[N];
		alignas(16) float vValues[N];

		memset(hits, 1, sizeof(hits));

#ifdef __INTEL_COMPILER
#pragma vector always assert aligned
#endif
		for (uint32_t i = 0; i < N; ++i)
		{
			const float v0v1X = vertex2X[i] - vertex1X[i];
			const float v0v1Y = vertex2Y[i] - vertex1Y[i];
			const float v0v1Z = vertex2Z[i] - vertex1Z[i];

			const float v0v2X = vertex3X[i] - vertex1X[i];
			const float v0v2Y = vertex3Y[i] - vertex1Y[i];
			const float v0v2Z = vertex3Z[i] - vertex1Z[i];

			// cross product
			const float pvecX = directionY * v0v2Z - directionZ * v0v2Y;
			const float pvecY = directionZ * v0v2X - directionX * v0v2Z;
			const float pvecZ = directionX * v0v2Y - directionY * v0v2X;

			// dot product
			const float determinant = v0v1X * pvecX + v0v1Y * pvecY + v0v1Z * pvecZ;

			if (std::abs(determinant) < std::numeric_limits<float>::epsilon())
				hits[i] = 0;

			const float invDeterminant = 1.0f / determinant;

			const float tvecX = originX - vertex1X[i];
			const float tvecY = originY - vertex1Y[i];
			const float tvecZ = originZ - vertex1Z[i];

			// dot product
			const float u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDeterminant;

			if (u < 0.0f || u > 1.0f)
				hits[i] = 0;

			// cross product
			const float qvecX = tvecY * v0v1Z - tvecZ * v0v1Y;
			const float qvecY = tvecZ * v0v1X - tvecX * v0v1Z;
			const float qvecZ = tvecX * v0v1Y - tvecY * v0v1X;

			// dot product
			const float v = (directionX * qvecX + directionY * qvecY + directionZ * qvecZ) * invDeterminant;

			if (v < 0.0f || (u + v) > 1.0f)
				hits[i] = 0;

			const float t = (v0v2X * qvecX + v0v2Y * qvecY + v0v2Z * qvecZ) * invDeterminant;

			if (t < 0.0f)
				hits[i] = 0;

			if (t < minDistance || t > maxDistance)
				hits[i] = 0;

			if (t > intersectionDistance)
				hits[i] = 0;

			uValues[i] = u;
			vValues[i] = v;
			distances[i] = t;
		}

		float distance, u, v;
		uint32_t triangleIndex;

		// if this isn't in its own function, the icc vectorizer gets confused
		if (!findIntersectionValues<N>(hits, distances, uValues, vValues, triangleIndices, distance, u, v, triangleIndex))
			return false;

		const Triangle& triangle = scene.bvhData.triangles[triangleIndex];

		if (ray.isShadowRay && triangle.material->nonShadowing)
			return false;

		return calculateIntersectionData(scene, ray, triangle, intersection, distance, u, v);
	}

	template <uint64_t N>
	bool Triangle::findIntersectionValues(const uint32_t* hits, const float* distances, const float* uValues, const float* vValues, const uint32_t* triangleIndices, float& distance, float& u, float& v, uint32_t& triangleIndex)
	{
		bool wasFound = false;
		distance = std::numeric_limits<float>::max();

		for (uint32_t i = 0; i < N; ++i)
		{
			if (hits[i] && distances[i] < distance)
			{
				wasFound = true;

				u = uValues[i];
				v = vValues[i];
				distance = distances[i];
				triangleIndex = triangleIndices[i];
			}
		}

		return wasFound;
	}
}

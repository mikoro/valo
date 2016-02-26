// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

#include "Tracing/AABB.h"

namespace Raycer
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	enum class BVHType { BVH1, BVH4 };

	struct BVHBuildTriangle
	{
		Triangle* triangle;
		AABB aabb;
		Vector3 center;
	};

	struct BVHSplitCache
	{
		AABB aabb;
		float cost;
	};

	struct BVHSplitOutput
	{
		uint64_t index;
		uint64_t axis;
		AABB fullAABB;
		AABB leftAABB;
		AABB rightAABB;
	};

	struct BVHNode
	{
		AABB aabb;
		int32_t rightOffset;
		uint32_t triangleOffset;
		uint32_t triangleCount;
		uint32_t splitAxis;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(triangleOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(splitAxis));
		}
	};

	template <uint64_t N>
	struct BVHNodeSOA
	{
		std::array<float, N> aabbMinX;
		std::array<float, N> aabbMinY;
		std::array<float, N> aabbMinZ;
		std::array<float, N> aabbMaxX;
		std::array<float, N> aabbMaxY;
		std::array<float, N> aabbMaxZ;
		std::array<uint32_t, N-1> rightOffset;
		std::array<uint16_t, N-1> splitAxis;
		uint32_t triangleOffset;
		uint32_t triangleCount;
		uint32_t isLeaf;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabbMinX),
				CEREAL_NVP(aabbMinY),
				CEREAL_NVP(aabbMinZ),
				CEREAL_NVP(aabbMaxX),
				CEREAL_NVP(aabbMaxY),
				CEREAL_NVP(aabbMaxZ),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(triangleOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(isLeaf));
		}
	};

	class BVH
	{
	public:

		virtual ~BVH() {}

		virtual void build(Scene& scene) = 0;
		virtual bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const = 0;

		static std::unique_ptr<BVH> getBVH(BVHType type);

	protected:

		static BVHSplitOutput calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end);
	};
}

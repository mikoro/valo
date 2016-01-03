// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/AABB.h"

namespace Raycer
{
	enum class BVHAxisSelection { LARGEST, RANDOM };
	enum class BVHAxisSplit { MIDDLE, MEDIAN, RANDOM };

	struct BVHBuildInfo
	{
		uint64_t maxLeafSize = 5;
		bool useSAH = true;
		uint64_t regularSAHSplits = 0;
		BVHAxisSelection axisSelection = BVHAxisSelection::LARGEST;
		BVHAxisSplit axisSplit = BVHAxisSplit::MEDIAN;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(maxLeafSize),
				CEREAL_NVP(useSAH),
				CEREAL_NVP(regularSAHSplits),
				CEREAL_NVP(axisSelection),
				CEREAL_NVP(axisSplit));
		}
	};

	struct BVHNode
	{
		AABB aabb;
		int64_t rightOffset;
		uint64_t startOffset;
		uint64_t primitiveCount;
		uint64_t splitAxis;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(startOffset),
				CEREAL_NVP(primitiveCount),
				CEREAL_NVP(splitAxis));
		}
	};

	struct BVHBuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
	};

	class Ray;
	class Intersection;
	class Triangle;
	class Scene;
	class Random;

	class BVH
	{
	public:

		bool intersect(const Ray& ray, Intersection& intersection) const;
		
		void build(const std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo);
		void restore(const Scene& scene);

		bool hasBeenBuilt() const;

	private:

		void calculateSplit(uint64_t& splitAxis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry, Random& random);
		void calculateSAHSplit(uint64_t& splitAxis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry);
		double calculateSAHScore(uint64_t splitAxis, double splitPoint, const AABB& nodeAABB, const BVHBuildEntry& buildEntry);
		double calculateMedianPoint(uint64_t splitAxis, const BVHBuildEntry& buildEntry);

		bool bvhHasBeenBuilt = false;
		std::vector<BVHNode> nodes;
		std::vector<uint64_t> orderedTriangleIds;
		std::vector<const Triangle*> orderedTriangles;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(bvhHasBeenBuilt),
				CEREAL_NVP(nodes),
				CEREAL_NVP(orderedTriangleIds));
		}
	};
}

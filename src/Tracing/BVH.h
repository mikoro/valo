// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/AABB.h"

namespace Raycer
{
	struct BVHBuildInfo
	{
		uint64_t maxLeafSize = 5;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(maxLeafSize));
		}
	};

	struct BVHNode
	{
		AABB aabb;
		int64_t rightOffset;
		uint64_t startOffset;
		uint64_t triangleCount;
		uint64_t splitAxis;
		uint8_t leftEnabled;
		uint8_t rightEnabled;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(startOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(splitAxis),
				CEREAL_NVP(leftEnabled),
				CEREAL_NVP(rightEnabled));
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

		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const;
		void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo);
		bool hasBeenBuilt() const;

		void disableLeft();
		void disableRight();
		void revertDisable();

	private:

		void calculateSplit(std::vector<Triangle>& triangles, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry);

		bool bvhHasBeenBuilt = false;
		std::vector<BVHNode> nodes;

		uint64_t disableIndex = 0;
		std::vector<uint64_t> previousDisableIndices;
		
		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(bvhHasBeenBuilt),
				CEREAL_NVP(nodes),
				CEREAL_NVP(disableIndex),
				CEREAL_NVP(previousDisableIndices));
		}
	};
}

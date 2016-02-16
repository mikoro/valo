// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

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

	struct BVHBuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
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

	class Triangle;
	class BVH;

	class BVHBuilder
	{
	public:

		BVHBuilder();

		void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo, BVH& bvh, std::atomic<bool>& interrupted);
		uint64_t getProcessedTrianglesCount() const;

	private:

		void calculateSplit(std::vector<Triangle>& triangles, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry);

		std::atomic<uint64_t> processedTrianglesCount;
	};
}

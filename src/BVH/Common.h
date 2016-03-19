// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/AABB.h"

namespace Raycer
{
	class Triangle;

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
	};
}

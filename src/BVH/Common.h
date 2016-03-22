// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>

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
		uint32_t index;
		uint32_t axis;
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

	template <uint32_t N>
	struct BVHNodeSOA
	{
		float aabbMinX[N];
		float aabbMinY[N];
		float aabbMinZ[N];
		float aabbMaxX[N];
		float aabbMaxY[N];
		float aabbMaxZ[N];
		uint32_t rightOffset[N-1];
		uint16_t splitAxis[N-1];
		uint32_t triangleOffset;
		uint32_t triangleCount;
		uint32_t isLeaf;
	};
}

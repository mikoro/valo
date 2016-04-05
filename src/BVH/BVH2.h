// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cstdint>
#include <vector>

#include "BVH/Common.h"
#include "Core/Common.h"
#include "Core/CudaAlloc.h"

namespace Raycer
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	class BVH2
	{
	public:

		BVH2();

		void build(std::vector<Triangle>& triangles);
		CUDA_CALLABLE bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

		uint32_t maxLeafSize = 4;

	private:

		CudaAlloc<BVHNode> nodesAlloc;
	};
}

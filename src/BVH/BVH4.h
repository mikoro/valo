// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "BVH/Common.h"
#include "Core/Common.h"
#include "Core/Triangle.h"
#include "Utils/CudaAlloc.h"

namespace Valo
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	class BVH4
	{
	public:

		BVH4();

		void build(std::vector<Triangle>& triangles);
		CUDA_CALLABLE bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

	private:

		CudaAlloc<BVHNodeSOA<4>> nodesAlloc;
		CudaAlloc<TriangleSOA<4>> triangles4Alloc;
	};
}

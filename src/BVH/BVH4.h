// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "BVH/Common.h"
#include "Core/Triangle.h"

namespace Raycer
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	class BVH4
	{
	public:

		~BVH4();

		void build(std::vector<Triangle>& triangles);
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

	private:

		BVHNodeSOA<4>* nodesPtr = nullptr;
		TriangleSOA<4>* triangles4Ptr = nullptr;
	};
}

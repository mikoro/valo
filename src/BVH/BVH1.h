// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/Common.h"

namespace Raycer
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	class BVH1
	{
	public:

		~BVH1();

		void build(std::vector<Triangle>& triangles);
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

		uint32_t maxLeafSize = 4;

	private:

		BVHNode* nodesPtr = nullptr;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(maxLeafSize));
		}
	};
}

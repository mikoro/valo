// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/BVH1.h"
#include "BVH/BVH4.h"
#include "BVH/Common.h"

namespace Raycer
{
	class Scene;
	class Ray;
	class Intersection;

	enum class BVHType { BVH1, BVH4 };

	class BVH
	{
	public:

		void build(std::vector<Triangle>& triangles, std::vector<TriangleSOA<4>>& triangles4);
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

		static BVHSplitOutput calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end);

		BVHType type = BVHType::BVH1;

		BVH1 bvh1;
		BVH4 bvh4;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(type),
				CEREAL_NVP(type),
				//CEREAL_NVP(bvh4),
				CEREAL_NVP(bvh1));
		}
	};
}

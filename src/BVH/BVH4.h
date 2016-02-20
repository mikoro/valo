// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/BVH.h"
#include "Tracing/AABB.h"

namespace Raycer
{
	struct BVH4Node
	{
		std::array<AABB, 4> aabb;
		std::array<uint64_t, 3> rightOffset;
		uint64_t triangleOffset;
		uint64_t triangleCount;
		bool isLeaf;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(triangleOffset),
				CEREAL_NVP(triangleCount));
		}
	};

	class BVH4 : public BVH
	{
	public:

		void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo) override;
		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const override;

	private:

		std::vector<BVH4Node> nodes;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(built),
				CEREAL_NVP(nodes));
		}
	};
}

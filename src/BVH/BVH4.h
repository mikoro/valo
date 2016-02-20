// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include <boost/align/aligned_allocator.hpp>

#include "cereal/cereal.hpp"

#include "BVH/BVH.h"
#include "Tracing/AABB.h"

namespace Raycer
{
	struct BVH4Node
	{
		std::array<float, 4> aabbMinX;
		std::array<float, 4> aabbMinY;
		std::array<float, 4> aabbMinZ;
		std::array<float, 4> aabbMaxX;
		std::array<float, 4> aabbMaxY;
		std::array<float, 4> aabbMaxZ;
		std::array<uint32_t, 3> rightOffset;
		uint32_t triangleOffset;
		uint32_t triangleCount;
		uint32_t isLeaf;
		uint32_t pad[2];

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabbMinX),
				CEREAL_NVP(aabbMinY),
				CEREAL_NVP(aabbMinZ),
				CEREAL_NVP(aabbMaxX),
				CEREAL_NVP(aabbMaxY),
				CEREAL_NVP(aabbMaxZ),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(triangleOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(isLeaf));
		}
	};

	class BVH4 : public BVH
	{
	public:

		void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo) override;
		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const override;

	private:

		std::vector<BVH4Node, boost::alignment::aligned_allocator<BVH4Node, 128>> nodes;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(built),
				CEREAL_NVP(nodes));
		}
	};
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include <boost/align/aligned_allocator.hpp>

#include "cereal/cereal.hpp"

#include "BVH/BVH.h"
#include "Tracing/Aabb.h"

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
		std::array<uint16_t, 3> splitAxis;
		uint32_t triangleOffset;
		uint32_t triangleCount;
		uint32_t isLeaf;

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

		void build(std::vector<Triangle>& triangles, uint64_t maxLeafSize) override;
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const override;

	private:

		std::vector<BVH4Node, boost::alignment::aligned_allocator<BVH4Node, 16>> nodes;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(nodes));
		}
	};
}

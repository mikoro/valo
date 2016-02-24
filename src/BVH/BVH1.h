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
	struct BVH1Node
	{
		AABB aabb;
		int64_t rightOffset;
		uint64_t startOffset;
		uint64_t triangleCount;
		uint64_t splitAxis;
		uint64_t pad;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(startOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(splitAxis));
		}
	};

	class BVH1 : public BVH
	{
	public:

		void build(std::vector<Triangle>& triangles, uint64_t maxLeafSize) override;
		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const override;

	private:

		std::vector<BVH1Node, boost::alignment::aligned_allocator<BVH1Node, 128>> nodes;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(nodes));
		}
	};
}

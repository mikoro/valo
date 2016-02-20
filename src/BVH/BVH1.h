// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/BVH.h"
#include "Tracing/AABB.h"

namespace Raycer
{
	class Vector2;

	struct BVH1Node
	{
		AABB aabb;
		int64_t rightOffset;
		uint64_t startOffset;
		uint64_t triangleCount;
		uint64_t splitAxis;
		uint8_t leftEnabled;
		uint8_t rightEnabled;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(aabb),
				CEREAL_NVP(rightOffset),
				CEREAL_NVP(startOffset),
				CEREAL_NVP(triangleCount),
				CEREAL_NVP(splitAxis),
				CEREAL_NVP(leftEnabled),
				CEREAL_NVP(rightEnabled));
		}
	};

	class BVH1 : public BVH
	{
	public:

		void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo) override;
		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const override;

		void disableLeft() override;
		void disableRight() override;
		void undoDisable() override;

	private:

		std::vector<BVH1Node> nodes;

		uint64_t disableIndex = 0;
		std::vector<uint64_t> previousDisableIndices;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(built),
				CEREAL_NVP(nodes));
		}
	};
}

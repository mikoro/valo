// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

namespace Raycer
{
	class Triangle;
	class Ray;
	class Intersection;

	enum class BVHType { BVH1, BVH4, BVH8, SBVH1 };

	struct BVHBuildInfo
	{
		uint64_t maxLeafSize = 5;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(maxLeafSize));
		}
	};

	struct BVHBuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
	};

	struct BVHSplitInput
	{
		std::vector<Triangle*>* trianglePtrs;
		std::vector<float>* rightScores;
		uint64_t startIndex;
		uint64_t endIndex;
		float nodeSurfaceArea;
	};

	struct BVHSplitOutput
	{
		uint64_t splitIndex = 0;
		uint64_t splitAxis = 0;
	};

	class BVH
	{
	public:

		virtual ~BVH() {}

		virtual bool hasBeenBuilt();
		virtual void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo) = 0;
		virtual bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const = 0;

		virtual void disableLeft();
		virtual void disableRight();
		virtual void undoDisable();

		static std::unique_ptr<BVH> getBVH(BVHType type);

	protected:

		static BVHSplitOutput calculateSplit(const BVHSplitInput& input);

		bool built = false;
	};
}

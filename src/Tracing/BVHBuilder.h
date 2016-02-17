// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
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

	class Triangle;
	class BVH;
	struct BVHNode;

	class BVHBuilder
	{
	public:

		static void build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo, BVH& bvh);

	private:

		static void calculateSplit(std::vector<Triangle*>& trianglePtrs, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry, std::vector<double>& rightScores);
	};
}

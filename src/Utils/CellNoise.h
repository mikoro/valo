// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Math/Color.h"

/*

http://graphics.ucsd.edu/courses/cse168_s06/ucsd/cellular_noise.pdf

Returns distance to closest neighbour (non-normalized) 0.0 - inf

*/

namespace Raycer
{
	const int CELL_NOISE_MAX_DISTANCES_COUNT = 100;

	enum class CellNoiseDistanceType { EUCLIDEAN, EUCLIDEAN_SQUARED, MANHATTAN, CHEBYSHEV };
	enum class CellNoiseCombineType { D1, D2, D1_PLUS_D2, D1_MINUS_D2, D1_TIMES_D2, D2_MINUS_D1 };

	class Vector3;

	class CellNoise
	{
	public:

		CellNoise();
		CellNoise(unsigned seed);

		void seed(unsigned seed);

		double getNoise(CellNoiseDistanceType distanceType, CellNoiseCombineType combineType, unsigned density, double x, double y, double z) const;
		double getNoise2D(CellNoiseDistanceType distanceType, CellNoiseCombineType combineType, unsigned density, double x, double y) const;

		void setVoronoiColors(const std::vector<Color>& colors);
		Color getVoronoiColor(CellNoiseDistanceType distanceType, unsigned density, double x, double y, double z) const;
		Color getVoronoiColor2D(CellNoiseDistanceType distanceType, unsigned density, double x, double y) const;

	private:

		int getHashcode(int x, int y, int z) const;

		double getDistance(CellNoiseDistanceType distanceType, const Vector3& v1, const Vector3& v2) const;
		double getCombinedValue(CellNoiseCombineType combineType, double d1, double d2) const;

		int m_seed;
		std::vector<Color> voronoiColors;
	};
}

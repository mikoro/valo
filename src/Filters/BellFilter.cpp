// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/BellFilter.h"

using namespace Raycer;

namespace
{
	float calculateWeight(float s)
	{
		s = std::abs(s);

		if (s < 0.5f)
			return 0.75f - (s * s);
		else if (s <= 1.5f)
			return 0.5f * std::pow(s - 1.5f, 2.0f);
		else
			return 0.0f;
	}
}

BellFilter::BellFilter()
{
	radiusX = 1.5;
	radiusY = 1.5;
}

float BellFilter::getWeightX(float x)
{
	return calculateWeight(x);
}

float BellFilter::getWeightY(float y)
{
	return calculateWeight(y);
}

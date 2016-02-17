// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/LanczosSincFilter.h"

using namespace Raycer;

namespace
{
	float sinc(float x)
	{
		return std::sin(float(M_PI) * x) / (float(M_PI) * x);
	}

	float calculateWeight(float s, float size)
	{
		s = std::abs(s);

		if (s == 0.0f)
			return 1.0f;

		if (s > size)
			return 0.0f;

		return sinc(s) * sinc(s / size);
	}
}

LanczosSincFilter::LanczosSincFilter(uint64_t radiusX_, uint64_t radiusY_)
{
	setRadius(radiusX_, radiusY_);
}

void LanczosSincFilter::setRadius(uint64_t radiusX_, uint64_t radiusY_)
{
	radiusX = float(radiusX_);
	radiusY = float(radiusY_);
}

float LanczosSincFilter::getWeightX(float x)
{
	return calculateWeight(x, radiusX);
}

float LanczosSincFilter::getWeightY(float y)
{
	return calculateWeight(y, radiusY);
}

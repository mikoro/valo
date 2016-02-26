// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/TentFilter.h"

using namespace Raycer;

namespace
{
	float calculateWeight(float s)
	{
		s = std::abs(s);

		if (s < 1.0f)
			return 1.0f - s;
		else
			return 0.0f;
	}
}

TentFilter::TentFilter(float radiusX_, float radiusY_)
{
	setRadius(radiusX_, radiusY_);
}

void TentFilter::setRadius(float radiusX_, float radiusY_)
{
	radiusX = radiusX_;
	radiusY = radiusY_;
	radiusXInv = 1.0f / radiusX;
	radiusYInv = 1.0f / radiusY;
}

float TentFilter::getWeightX(float x)
{
	return calculateWeight(x * radiusXInv) * radiusXInv;
}

float TentFilter::getWeightY(float y)
{
	return calculateWeight(y * radiusYInv) * radiusYInv;
}

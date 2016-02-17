// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/BoxFilter.h"

using namespace Raycer;

BoxFilter::BoxFilter(float radiusX_, float radiusY_)
{
	setRadius(radiusX_, radiusY_);
}

void BoxFilter::setRadius(float radiusX_, float radiusY_)
{
	radiusX = radiusX_;
	radiusY = radiusY_;
	weightX = 1.0f / (2.0f * radiusX);
	weightY = 1.0f / (2.0f * radiusY);
}

float BoxFilter::getWeightX(float x)
{
	if (x >= -radiusX && x < radiusX)
		return weightX;
	else
		return 0.0f;
}

float BoxFilter::getWeightY(float y)
{
	if (y >= -radiusY && y < radiusY)
		return weightY;
	else
		return 0.0f;
}

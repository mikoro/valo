// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Filters/BoxFilter.h"

using namespace Raycer;

namespace
{
	CUDA_CALLABLE float calculateWeight(float s, float radius)
	{
		float weight = 1.0f / (2.0f * radius);
		return (s >= -radius && s < radius) ? weight : 0.0f;
	}
}

CUDA_CALLABLE float BoxFilter::getWeight(float s)
{
	return calculateWeight(s, radius.x);
}

CUDA_CALLABLE float BoxFilter::getWeight(const Vector2& point)
{
	return calculateWeight(point.x, radius.x) * calculateWeight(point.y, radius.y);
}

CUDA_CALLABLE Vector2 BoxFilter::getRadius()
{
	return radius;
}

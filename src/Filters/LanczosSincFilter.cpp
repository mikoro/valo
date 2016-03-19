// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

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

float LanczosSincFilter::getWeight(float s)
{
	return calculateWeight(s, radius.x);
}

float LanczosSincFilter::getWeight(const Vector2& point)
{
	return calculateWeight(point.x, radius.x) * calculateWeight(point.y, radius.y);
}

Vector2 LanczosSincFilter::getRadius()
{
	return radius;
}

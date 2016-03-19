// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

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

float TentFilter::getWeight(float s)
{
	float radiusInv = 1.0f / radius.x;

	return calculateWeight(s * radiusInv) * radiusInv;
}

float TentFilter::getWeight(const Vector2& point)
{
	float radiusXInv = 1.0f / radius.x;
	float radiusYInv = 1.0f / radius.y;

	return calculateWeight(point.x * radiusXInv) * radiusXInv * calculateWeight(point.y * radiusYInv) * radiusYInv;
}

Vector2 TentFilter::getRadius()
{
	return radius;
}

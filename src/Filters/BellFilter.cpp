// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Filters/BellFilter.h"
#include "Math/Vector2.h"

using namespace Raycer;

namespace
{
	float calculateWeight(float s)
	{
		s = std::abs(s);

		if (s < 0.5f)
			return 0.75f - (s * s);
		
		if (s <= 1.5f)
			return 0.5f * std::pow(s - 1.5f, 2.0f);
		
		return 0.0f;
	}
}

float BellFilter::getWeight(float s)
{
	return calculateWeight(s);
}

float BellFilter::getWeight(const Vector2& point)
{
	return calculateWeight(point.x) * calculateWeight(point.y);
}

Vector2 BellFilter::getRadius()
{
	return Vector2(1.5f, 1.5f);
}

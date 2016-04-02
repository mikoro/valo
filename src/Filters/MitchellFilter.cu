// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/MitchellFilter.h"

using namespace Raycer;

namespace
{
	CUDA_CALLABLE float calculateWeight(float s, float B, float C)
	{
		s = std::abs(s);

		if (s <= 1.0f)
			return ((12.0f - 9.0f * B - 6.0f * C) * (s * s * s) + (-18.0f + 12.0f * B + 6.0f * C) * (s * s) + (6.0f - 2.0f * B)) * (1.0f / 6.0f);
		
		if (s <= 2.0f)
			return ((-B - 6.0f * C) * (s * s * s) + (6.0f * B + 30.0f * C) * (s * s) + (-12.0f * B - 48.0f * C) * s + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
		
		return 0.0f;
	}
}

CUDA_CALLABLE float MitchellFilter::getWeight(float s)
{
	return calculateWeight(s, B, C);
}

CUDA_CALLABLE float MitchellFilter::getWeight(const Vector2& point)
{
	return calculateWeight(point.x, B, C) * calculateWeight(point.y, B, C);
}

CUDA_CALLABLE Vector2 MitchellFilter::getRadius()
{
	return Vector2(2.0f, 2.0f);
}

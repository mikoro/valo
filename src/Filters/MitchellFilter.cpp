// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/MitchellFilter.h"

using namespace Raycer;

namespace
{
	float calculateWeight(float s, float B, float C)
	{
		s = std::abs(s);

		if (s <= 1.0f)
			return ((12.0f - 9.0f * B - 6.0f * C) * (s * s * s) + (-18.0f + 12.0f * B + 6.0f * C) * (s * s) + (6.0f - 2.0f * B)) * (1.0f / 6.0f);
		else if (s <= 2.0f)
			return ((-B - 6.0f * C) * (s * s * s) + (6.0f * B + 30.0f * C) * (s * s) + (-12.0f * B - 48.0f * C) * s + (8.0f * B + 24.0f * C)) * (1.0f / 6.0f);
		else
			return 0.0f;
	}
}

MitchellFilter::MitchellFilter(float B_, float C_)
{
	setCoefficients(B_, C_);
}

void MitchellFilter::setCoefficients(float B_, float C_)
{
	B = B_;
	C = C_;
	radiusX = 2.0f;
	radiusY = 2.0f;
}

float MitchellFilter::getWeightX(float x)
{
	return calculateWeight(x, B, C);
}

float MitchellFilter::getWeightY(float y)
{
	return calculateWeight(y, B, C);
}

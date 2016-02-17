// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <complex>
#include <limits>

namespace Raycer
{
	class MathUtils
	{
	public:

		static bool almostZero(float value, float threshold = std::numeric_limits<float>::epsilon() * 4);
		static bool almostSame(float first, float second, float threshold = std::numeric_limits<float>::epsilon() * 4);
		static bool almostSame(const std::complex<float>& first, const std::complex<float>& second, float threshold = std::numeric_limits<float>::epsilon() * 4);
		static float degToRad(float degrees);
		static float radToDeg(float radians);
		static float smoothstep(float t);
		static float smootherstep(float t);
		static float fastPow(float a, float b);
	};
}

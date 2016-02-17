// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/Solver.h"

using namespace Raycer;

// numerically stable quadratic formula
QuadraticResult Solver::findQuadraticRoots(float a, float b, float c)
{
	QuadraticResult result;

	float discriminant = b * b - 4.0f * a * c;

	if (discriminant < 0.0f)
		return result;

	float q = -0.5f * (b + std::copysign(1.0f, b) * std::sqrt(discriminant));

	result.roots[0] = q / a;
	result.roots[1] = c / q;
	result.rootCount = (result.roots[0] == result.roots[1]) ? 1 : 2;

	// order ascending
	if (result.roots[0] > result.roots[1])
		std::swap(result.roots[0], result.roots[1]);

	return result;
}

// false position / regula falsi
// https://en.wikipedia.org/wiki/False_position_method
float Solver::findRoot(const std::function<float(float)>& f, float begin, float end, uint64_t iterations)
{
	float x0 = 0.0f;
	float y0 = 0.0f;
	float x1 = begin;
	float y1 = f(x1);
	float x2 = end;
	float y2 = f(x2);

	for (uint64_t i = 0; i < iterations; ++i)
	{
		x0 = x1 - ((y1 * (x2 - x1)) / (y2 - y1));
		y0 = f(x0);

		if (y0 * y1 > 0.0f)
		{
			x1 = x0;
			y1 = y0;
		}
		else
		{
			x2 = x0;
			y2 = y0;
		}
	}

	return x0;
}

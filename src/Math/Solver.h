// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <functional>

namespace Valo
{
	struct QuadraticResult
	{
		uint32_t rootCount = 0;
		float roots[2];
	};

	class Solver
	{
	public:

		static QuadraticResult findQuadraticRoots(float a, float b, float c);
		static float findRoot(const std::function<float(float)>& f, float begin, float end, uint32_t iterations = 32);
	};
}

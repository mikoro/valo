// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Math/Solver.h"
#include "Math/MathUtils.h"

using namespace Raycer;

TEST_CASE("Solver functionality", "[solver]")
{
	QuadraticResult result1 = Solver::findQuadraticRoots(1.5f, 4.0f, -3.0f);

	REQUIRE(result1.rootCount == 2);
	REQUIRE(MathUtils::almostSame(result1.roots[0], -3.2769839649484336f));
	REQUIRE(MathUtils::almostSame(result1.roots[1], 0.61031729828176684f));

	auto f1 = [](float x) { return std::cos(x) - x * x * x; };
	float result2 = Solver::findRoot(f1, -2.0f, 2.0f, 32);

	REQUIRE(MathUtils::almostSame(result2, 0.86547378345385151f));
}

#endif

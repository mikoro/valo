// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Math/Polynomial.h"
#include "Math/MathUtils.h"

using namespace Raycer;

TEST_CASE("Polynomial functionality", "[polynomial]")
{
	Polynomial<5> polynomial;

	float coefficients[5] = { 2.0f, 7.0f, 5.0f, 3.0f, -10.0f };
	polynomial.setCoefficients(coefficients);

	uint64_t count;
	const float* result = polynomial.findAllPositiveRealRoots(count, 32, 0.0001f);

	REQUIRE(count == 1);
	REQUIRE(MathUtils::almostSame(result[0], 0.79988784795406020f));
}

#endif

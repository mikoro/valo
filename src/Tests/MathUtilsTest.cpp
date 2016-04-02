// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Math/MathUtils.h"

using namespace Raycer;

TEST_CASE("MathUtils functionality", "[mathutils]")
{
	const float epsilon = FLT_EPSILON;
	const float min = FLT_MIN;

	REQUIRE(MathUtils::almostZero(0.0f, epsilon) == true);
	REQUIRE(MathUtils::almostZero(0.0001f, epsilon) == false);
	REQUIRE(MathUtils::almostZero(-0.0001f, epsilon) == false);
	REQUIRE(MathUtils::almostZero(epsilon / 2, epsilon) == true);
	REQUIRE(MathUtils::almostZero(epsilon, 2 * epsilon) == true);
	REQUIRE(MathUtils::almostZero(epsilon * 2, epsilon) == false);
	REQUIRE(MathUtils::almostZero(min, epsilon) == true);
	REQUIRE(MathUtils::almostZero(-epsilon / 2, epsilon) == true);
	REQUIRE(MathUtils::almostZero(-epsilon, 2 * epsilon) == true);
	REQUIRE(MathUtils::almostZero(-epsilon * 2, epsilon) == false);
	REQUIRE(MathUtils::almostZero(-min) == true);

	REQUIRE(MathUtils::almostSame(epsilon, epsilon, epsilon) == true);
	REQUIRE(MathUtils::almostSame(epsilon, epsilon * 2, epsilon) == false);
	REQUIRE(MathUtils::almostSame(epsilon, epsilon * 2, 10 * epsilon) == true);

	REQUIRE(MathUtils::almostSame(MathUtils::degToRad(90.0f), float(M_PI) / 2.0f));
	REQUIRE(MathUtils::almostSame(MathUtils::degToRad(180.0f), float(M_PI)));
}

#endif

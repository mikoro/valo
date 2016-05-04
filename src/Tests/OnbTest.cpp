// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Math/ONB.h"

using namespace Raycer;

TEST_CASE("ONB functionality", "[onb]")
{
	ONB onb1 = ONB::fromNormal(Vector3(0.0f, 0.0f, -1.0f), Vector3::up());

	REQUIRE(onb1.u == Vector3(1.0f, 0.0f, 0.0f));
	REQUIRE(onb1.v == Vector3(0.0f, 1.0f, 0.0f));
	REQUIRE(onb1.w == Vector3(0.0f, 0.0f, -1.0f));
}

#endif

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Core/ONB.h"

using namespace Raycer;

TEST_CASE("ONB functionality", "[onb]")
{
	ONB onb1 = ONB::fromNormal(Vector3(0.0, 0.0, -1.0), Vector3::UP);

	REQUIRE(onb1.u == Vector3(1.0, 0.0, 0.0));
	REQUIRE(onb1.v == Vector3(0.0, 1.0, 0.0));
	REQUIRE(onb1.w == Vector3(0.0, 0.0, -1.0));
}

#endif

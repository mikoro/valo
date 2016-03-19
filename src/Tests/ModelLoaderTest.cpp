// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Utils/ModelLoader.h"

using namespace Raycer;

TEST_CASE("ModelLoader functionality", "[modelloader]")
{
	ModelLoaderInfo info;
	ModelLoader modelLoader;

	info.modelFileName = "data/models/cornellbox-cuboids/cornellbox.obj";
	ModelLoaderResult result = modelLoader.load(info);
	
	REQUIRE(result.triangles.size() == 158);
}

#endif

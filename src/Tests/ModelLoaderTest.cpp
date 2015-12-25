// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Utils/ModelLoader.h"

using namespace Raycer;

TEST_CASE("ModelLoader functionality", "[modelloader]")
{
	ModelLoaderInfo info;
	info.modelFilePath = "data/models/cornellbox/cornellbox.obj";
	ModelLoaderResult result = ModelLoader::load(info);
	REQUIRE(result.triangles.size() == 36);
}

#endif

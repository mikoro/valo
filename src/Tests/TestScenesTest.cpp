// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Core/Scene.h"
#include "TestScenes/TestScene.h"

using namespace Raycer;

TEST_CASE("TestScenes functionality", "[testscenes]")
{
	std::vector<Scene> scenes;

	scenes.push_back(TestScene::create1());
	scenes.push_back(TestScene::create2());
	scenes.push_back(TestScene::create3());
	scenes.push_back(TestScene::create4());
	scenes.push_back(TestScene::create5());
	scenes.push_back(TestScene::create6());
	scenes.push_back(TestScene::create7());
	scenes.push_back(TestScene::create8());
	scenes.push_back(TestScene::create9());

	uint64_t sceneCount = 0;

	for (const Scene& scene : scenes)
		scene.save(tfm::format("scene%d.xml", sceneCount++));
}

#endif

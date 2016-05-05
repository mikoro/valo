// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "App.h"
#include "Core/Scene.h"
#include "TestScenes/TestScene.h"
#include "Utils/Log.h"

using namespace Raycer;

Scene TestScene::create(uint32_t number)
{
	App::getLog().logInfo("Creating test scene number %d", number);

	switch (number)
	{
		case 1: return create1();
		case 2: return create2();
		case 3: return create3();
		case 4: return create4();
		case 5: return create5();
		case 6: return create6();
		default: throw std::runtime_error("Unknown test scene number");
	}
}

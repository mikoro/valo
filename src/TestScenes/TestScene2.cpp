// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CORNELL BOX SPHERES //

Scene TestScene::create2()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(0.0000f, 0.7808f, 2.9691f);

	ModelLoaderInfo model;
	model.modelFileName = "data/models/cornellbox-spheres/cornellbox.obj";

	scene.models.push_back(model);

	return scene;
}

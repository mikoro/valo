// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CUSTOM SPHERES SCENE //

Scene TestScene::create3()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(0.0f, 1.0f, 0.0f);
	scene.camera.orientation = EulerAngle(0.0f, 0.0f, 0.0f);

	ModelLoaderInfo model;
	model.modelFileName = "data/models/spheres/spheres.obj";

	scene.models.push_back(model);

	return scene;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// BEDROOM //

Scene TestScene::create9()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(0.0836f, 1.8613f, 2.6068f);
	scene.camera.orientation = EulerAngle(-11.3541f, 24.7832f, 0.0000f);

	// MODELS //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

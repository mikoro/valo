// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// BEDROOM //

Scene TestScene::create8()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(2.5747f, 1.5601f, 2.5999f);
	scene.camera.orientation = EulerAngle(-9.9896f, 47.4465f, 0.0000f);

	scene.bvh.type = BVHType::BVH4;

	// MODELS //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/bedroom/bedroom.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

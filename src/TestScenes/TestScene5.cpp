// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// KITCHEN //

Scene TestScene::create5()
{
	Scene scene;

	scene.bvh.type = BVHType::BVH2;

	scene.renderer.imageSamples = 1;
	scene.renderer.pixelSamples = 1;

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(0.0836f, 1.8613f, 2.6068f);
	scene.camera.orientation = EulerAngle(-11.3541f, 24.7832f, 0.0000f);
	scene.camera.vignette = true;
	scene.camera.vignetteFactor1 = 1.0f;
	scene.camera.vignetteFactor2 = 0.1f;

	// MODELS //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

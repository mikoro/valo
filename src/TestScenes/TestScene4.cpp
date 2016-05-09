// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Valo;

// KITCHEN 3 //

Scene TestScene::create4()
{
	Scene scene;

	scene.bvh.type = BVHType::BVH2;

	scene.integrator.aoIntegrator.maxDistance = 0.5f;

	scene.integrator.type = IntegratorType::PATH;
	scene.renderer.filter.type = FilterType::MITCHELL;
	scene.tonemapper.type = TonemapperType::REINHARD;
	scene.tonemapper.reinhardTonemapper.key = 0.06f;

	scene.camera.position = Vector3(1.5705f, 1.3210f, 0.2419f);
	scene.camera.orientation = EulerAngle(-2.2362f, 69.1348f, 0.0000f);
	scene.camera.fishEyeAngle = 125.0f;
	scene.camera.type = CameraType::FISHEYE;

	// MODEL //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

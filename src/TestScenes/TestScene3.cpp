// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// KITCHEN 2 //

Scene TestScene::create3()
{
	Scene scene;

	scene.bvh.type = BVHType::BVH2;

	scene.integrator.aoIntegrator.maxDistance = 0.5f;

	scene.integrator.type = IntegratorType::PATH;
	scene.renderer.filter.type = FilterType::MITCHELL;
	scene.tonemapper.type = TonemapperType::REINHARD;
	scene.tonemapper.reinhardTonemapper.key = 0.25f;

	scene.camera.position = Vector3(0.4050f, 1.3779f, 1.2261f);
	scene.camera.orientation = EulerAngle(-25.2376f, 52.6524f, 0.0000f);
	scene.camera.depthOfField = true;
	scene.camera.focalDistance = 1.0f;
	scene.camera.apertureSize = 0.01f;

	// MODEL //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Valo;

// KITCHEN 1 //

Scene TestScene::create2()
{
	Scene scene;

	scene.bvh.type = BVHType::BVH2;

	scene.integrator.aoIntegrator.maxDistance = 0.5f;

	scene.integrator.type = IntegratorType::PATH;
	scene.renderer.filter.type = FilterType::MITCHELL;
	scene.tonemapper.type = TonemapperType::REINHARD;
	scene.tonemapper.reinhardTonemapper.key = 0.25f;

	scene.camera.position = Vector3(0.0836f, 1.8613f, 2.6068f);
	scene.camera.orientation = EulerAngle(-11.3541f, 24.7832f, 0.0000f);
	scene.camera.vignette = true;
	scene.camera.vignettePower = 1.0f;
	scene.camera.vignetteOffset = 0.1f;

	// MODEL //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

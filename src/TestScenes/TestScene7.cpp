// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CONFERENCE //

Scene TestScene::create7()
{
	Scene scene;

	scene.integrator.type = IntegratorType::AMBIENT_OCCLUSION;
	scene.integrator.aoIntegrator.maxDistance = 0.5f;
	scene.renderer.filter.type = FilterType::MITCHELL;
	scene.tonemapper.type = TonemapperType::LINEAR;
	scene.tonemapper.linearTonemapper.exposure = -0.5f;

	scene.camera.position = Vector3(-4.6401f, 0.4618f, 2.7327f);
	scene.camera.orientation = EulerAngle(-13.3503f, -56.3473f, 0.0000f);
	scene.camera.fov = 65.0f;

	scene.bvh.type = BVHType::BVH2;

	ModelLoaderInfo model;
	model.modelFileName = "data/models/conference/conference.obj";

	scene.models.push_back(model);

	return scene;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CORNELL BOX //

Scene TestScene::create1()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;
	scene.integrator.aoIntegrator.maxDistance = 0.5f;
	scene.bvh.type = BVHType::BVH2;

	scene.camera.position = Vector3(0.0f, 1.0f, 3.5f);

	scene.volume.enabled = false;
	scene.volume.attenuation = true;
	scene.volume.emission = false;
	scene.volume.inscatter = false;
	scene.volume.constant = true;
	scene.volume.stepSize = 0.1f;
	scene.volume.constantDensity = 0.5f;
	scene.volume.noisePositionScale = 10.0f;
	scene.volume.noiseValueScale = 0.5f;

	ModelLoaderInfo model;
	model.modelFileName = "data/models/cornellbox/cornellbox.obj";

	scene.models.push_back(model);

	return scene;
}

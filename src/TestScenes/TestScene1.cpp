﻿// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CORNELL BOX //

Scene TestScene::create1()
{
	Scene scene;

	scene.integrator.type = IntegratorType::PATH;

	scene.camera.position = Vector3(0.0f, 1.0f, 3.5f);

	scene.bvh.type = BVHType::BVH2;

	scene.integrator.aoIntegrator.maxDistance = 0.5f;

	scene.renderer.imageSamples = 1;
	scene.renderer.pixelSamples = 1;

	ModelLoaderInfo model;
	model.modelFileName = "data/models/cornellbox/cornellbox.obj";

	scene.models.push_back(model);

	return scene;
}

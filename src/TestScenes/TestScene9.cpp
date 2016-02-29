// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// BEDROOM //

Scene TestScene::create9()
{
	Scene scene;

	scene.general.tracerType = TracerType::PATH;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::MITCHELL;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(0.0836f, 1.8613f, 2.6068f);
	scene.camera.orientation = EulerAngle(-11.3541f, 24.7832f, 0.0000f);

	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	scene.bvhInfo.loadFromFile = false;
	scene.imagePoolInfo.loadFromFile = false;

	// MODELS //

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/kitchen/kitchen.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	return scene;
}

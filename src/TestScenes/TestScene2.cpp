// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// CORNELL BOX SPHERES //

Scene TestScene::create2()
{
	Scene scene;

	scene.general.tracerType = TracerType::PATH;
	
	scene.raytracing.maxIterationDepth = 6;

	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::BELL;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.2f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(-0.0000f, 0.7808f, 2.9691f);
	
	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/cornellbox-spheres/cornellbox.obj";

	scene.models.push_back(model);

	return scene;
}

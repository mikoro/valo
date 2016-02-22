// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// CUSTOM SPHERES SCENE //

Scene TestScene::create3()
{
	Scene scene;

	scene.general.tracerType = TracerType::RAY;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::BELL;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.2f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(0.0f, 1.0f, 0.0f);

	scene.bvhType = BVHType::BVH1;
	scene.bvhBuildInfo.maxLeafSize = 4;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/spheres/spheres.obj";

	scene.models.push_back(model);
	
	AmbientLight ambientLight;
	ambientLight.color = Color(1.0f, 1.0f, 1.0f) * 0.01f;
	ambientLight.occlusion = false;
	ambientLight.maxSampleDistance = 0.2f;
	ambientLight.sampleCountSqrt = 4;

	scene.lights.ambientLights.push_back(ambientLight);

	PointLight pointLight;
	pointLight.color = Color(1.0f, 1.0f, 1.0f) * 2.0f;
	pointLight.position = Vector3(0.0f, 3.9f, 0.0f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// BEDROOM //

Scene TestScene::create8()
{
	Scene scene;

	scene.general.tracerType = TracerType::RAY;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::TENT;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(2.5747f, 1.5601f, 2.5999f);
	scene.camera.orientation = EulerAngle(-9.9896f, 47.4465f, 0.0000f);

	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	scene.bvhInfo.loadFromFile = false;
	scene.imagePoolInfo.loadFromFile = false;

	// MODELS //

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/bedroom/bedroom.obj";
	model.scale = Vector3(1.0f, 1.0f, 1.0f);

	scene.models.push_back(model);

	// LIGHTS //

	AmbientLight ambientLight;
	ambientLight.color = Color::WHITE * 0.05f;

	scene.lights.ambientLights.push_back(ambientLight);
	
	PointLight pointLight;
	pointLight.color = Color::WHITE * 2.0f;
	pointLight.position = Vector3(0.0f, 2.0f, 2.0f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

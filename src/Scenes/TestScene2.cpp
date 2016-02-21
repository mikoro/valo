// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene2()
{
	Scene scene;

	scene.general.tracerType = TracerType::PREVIEW;

	scene.bvhType = BVHType::BVH1;
	scene.bvhBuildInfo.maxLeafSize = 4;

	scene.camera.position = Vector3(-4.6401f, 0.4618f, 2.7327f);
	scene.camera.orientation = EulerAngle(-13.3503f, -56.3473f, 0.0000f);
	scene.camera.fov = 65.0f;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/conference/conference.obj";

	scene.models.push_back(model);
	
	PointLight pointLight;
	pointLight.color = Color(1.0f, 1.0f, 1.0f) * 2.0f;
	pointLight.position = Vector3(-0.64f, 1.09f, -0.34f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

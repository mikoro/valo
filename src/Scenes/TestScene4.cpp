// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene4()
{
	Scene scene;

	scene.general.tracerType = TracerType::PREVIEW;

	scene.bvhType = BVHType::BVH1;
	scene.bvhBuildInfo.maxLeafSize = 4;

	scene.camera.position = Vector3(-0.9688f, 0.5192f, -0.9582f);
	scene.camera.orientation = EulerAngle(-17.5715f, -129.0079f, 0.0000f);
	scene.camera.fov = 82.0f;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/indirect-test/indirect-test_mat.obj";

	scene.models.push_back(model);
	
	PointLight pointLight;
	pointLight.color = Color(1.0f, 1.0f, 1.0f) * 2.0f;
	pointLight.position = Vector3(-0.64f, 1.09f, -0.34f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

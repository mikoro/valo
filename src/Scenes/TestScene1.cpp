// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stdafx.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene1()
{
	Scene scene;

	scene.camera.position = Vector3(0.0, 1.0, 3.5);

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/cornellbox/cornellbox.obj";

	scene.models.push_back(model);
	
	scene.lights.ambientLight.color = Color(1.0, 1.0, 1.0);
	scene.lights.ambientLight.intensity = 0.01;

	PointLight pointLight;
	pointLight.color = Color(1.0, 0.71, 0.24);
	pointLight.intensity = 1.0;
	pointLight.position = Vector3(0.0, 1.9, 0.0);
	pointLight.maxDistance = 3.0;
	pointLight.attenuation = 1.0;

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene2()
{
	Scene scene;

	scene.general.tracerType = TracerType::RAY;

	scene.camera.position = Vector3(3.14, 0.82, 2.50);
	scene.camera.orientation = EulerAngle(-10.91, 52.88, 0.0);

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/conference/conference.obj";

	scene.models.push_back(model);
	
	PointLight pointLight;
	pointLight.color = Color(1.0, 1.0, 1.0) * 2.0;
	pointLight.position = Vector3(-0.64, 1.09, -0.34);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

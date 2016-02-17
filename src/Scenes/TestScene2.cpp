// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene2()
{
	Scene scene;

	scene.general.tracerType = TracerType::RAY;

	scene.camera.position = Vector3(3.14f, 0.82f, 2.50f);
	scene.camera.orientation = EulerAngle(-10.91f, 52.88f, 0.0f);

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/conference/conference.obj";

	scene.models.push_back(model);
	
	PointLight pointLight;
	pointLight.color = Color(1.0f, 1.0f, 1.0f) * 2.0f;
	pointLight.position = Vector3(-0.64f, 1.09f, -0.34f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

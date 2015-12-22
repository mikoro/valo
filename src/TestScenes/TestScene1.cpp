// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stdafx.h"

#include "Tracing/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene1()
{
	Scene scene;

	scene.camera.position = Vector3(0.0, 1.0, 3.5);

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/cornellbox/cornellbox.obj";

	scene.models.push_back(model);

	scene.lights.ambientLight.intensity = 0.1;

	return scene;
}

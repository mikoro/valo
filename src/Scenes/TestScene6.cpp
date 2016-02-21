// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene6()
{
	Scene scene;

	scene.general.tracerType = TracerType::PREVIEW;

	scene.bvhType = BVHType::BVH1;
	scene.bvhBuildInfo.maxLeafSize = 4;

	scene.camera.position = Vector3(-6.1805f, 6.4027f, 3.4158f);
	scene.camera.orientation = EulerAngle(-14.6954f, -45.0006f, 0.0000f);
	scene.camera.fov = 12.0f;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/sponza_old/sponza_old.obj";
	model.defaultMaterialId = 1;

	scene.models.push_back(model);

	DiffuseSpecularMaterial material;
	material.id = 1;
	material.reflectance = Color::WHITE;

	scene.materials.diffuseSpecularMaterials.push_back(material);
	
	PointLight pointLight;
	pointLight.color = Color(1.0f, 1.0f, 1.0f) * 2.0f;
	pointLight.position = Vector3(-0.64f, 1.09f, -0.34f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

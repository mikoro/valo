// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene5()
{
	Scene scene;

	scene.general.tracerType = TracerType::PREVIEW;

	scene.bvhType = BVHType::BVH1;
	scene.bvhBuildInfo.maxLeafSize = 4;

	scene.camera.position = Vector3(-13.2988f, 7.5098f, -1.9199f);
	scene.camera.orientation = EulerAngle(-9.3740f, -105.8906f, 0.0000f);
	scene.camera.fov = 60.0f;

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

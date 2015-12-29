// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene3()
{
	Scene scene;

	scene.sampling.pixelSampleCount = 1;
	scene.general.tracerType = TracerType::RAY;

	scene.camera.position = Vector3(8.92, 0.68, -2.02);
	scene.camera.orientation = EulerAngle(6.66, 111.11, 0.0);

	// BOUNDING SPHERE MODEL //

	DiffuseSpecularMaterial sphereMaterial;
	sphereMaterial.id = 1;
	sphereMaterial.diffuseReflectance = Color(1.0, 1.0, 1.0);
	sphereMaterial.emittance = Color(1.0, 1.0, 1.0) * 8.0;
	sphereMaterial.skipLighting = true;
	sphereMaterial.nonShadowing = true;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/sphere.obj";
	model.defaultMaterialId = sphereMaterial.id;
	model.idStartOffset = 1;
	model.scale = Vector3(1000.0, 1000.0, 1000.0);

	scene.materials.diffuseSpecularMaterials.push_back(sphereMaterial);
	scene.models.push_back(model);

	// SPONZA MODEL //

	model = ModelLoaderInfo();
	model.modelFilePath = "data/models/sponza/sponza.obj";
	model.idStartOffset = 1000000;
	model.scale = Vector3(0.01, 0.01, 0.01);

	scene.models.push_back(model);
	
	PointLight pointLight;
	pointLight.color = Color(1.0, 1.0, 1.0) * 5.0;
	pointLight.position = Vector3(5.0, 8.0, 0.0);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

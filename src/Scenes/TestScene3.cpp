// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Scenes/Scene.h"

using namespace Raycer;

Scene Scene::createTestScene3()
{
	Scene scene;

	Color skyColor(182, 126, 91);

	scene.general.tracerType = TracerType::PATH_ITERATIVE;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::TENT;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(8.92, 0.68, -2.02);
	scene.camera.orientation = EulerAngle(6.66, 111.11, 0.0);

	// PLANE MODEL //

	DiffuseSpecularMaterial planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.diffuseReflectance = skyColor;
	planeMaterial.emittance = skyColor * 10.0;
	planeMaterial.invertNormal = false;
	planeMaterial.skipLighting = true;
	planeMaterial.nonShadowing = true;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/plane.obj";
	model.defaultMaterialId = planeMaterial.id;
	model.idStartOffset = 1;
	model.scale = Vector3(9.0, 1.0, 2.0);
	model.translate = Vector3(-0.6, 13.0, -0.3);
	model.rotate = EulerAngle(180.0, 0.0, 0.0);

	scene.materials.diffuseSpecularMaterials.push_back(planeMaterial);
	scene.models.push_back(model);

	// SPONZA MODEL //

	model = ModelLoaderInfo();
	model.modelFilePath = "data/models/sponza/sponza.obj";
	model.idStartOffset = 1000000;
	model.scale = Vector3(0.01, 0.01, 0.01);

	scene.models.push_back(model);

	// LIGHTS //

	AmbientLight ambientLight;
	ambientLight.color = skyColor * 0.05;

	scene.lights.ambientLights.push_back(ambientLight);
	
	PointLight pointLight;
	pointLight.color = skyColor * 200.0;
	pointLight.position = Vector3(0.0, 18.0, 0.0);

	scene.lights.pointLights.push_back(pointLight);

	DirectionalLight directionalLight;
	directionalLight.color = skyColor * 1.0;
	directionalLight.direction = EulerAngle(-80.0, 0.0, 0.0).getDirection();

	//scene.lights.directionalLights.push_back(directionalLight);

	return scene;
}

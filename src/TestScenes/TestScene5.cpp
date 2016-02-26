// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// DABROVIC SPONZA //

Scene TestScene::create5()
{
	Scene scene;

	Color skyColor(182, 126, 91);

	scene.general.tracerType = TracerType::RAY;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::TENT;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(8.92f, 0.68f, -2.02f);
	scene.camera.orientation = EulerAngle(6.66f, 111.11f, 0.0f);

	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	// PLANE MODEL //

	DiffuseSpecularMaterial planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.diffuseReflectance = skyColor;
	planeMaterial.emittance = skyColor * 10.0f;
	planeMaterial.invertNormal = false;
	planeMaterial.skipLighting = true;
	planeMaterial.nonShadowing = true;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/plane.obj";
	model.defaultMaterialId = planeMaterial.id;
	model.scale = Vector3(11.0f, 1.0f, 3.0f);
	model.translate = Vector3(0.0f, 15.7f, 0.0f);
	model.rotate = EulerAngle(180.0f, 0.0f, 0.0f);

	scene.materials.diffuseSpecularMaterials.push_back(planeMaterial);
	scene.models.push_back(model);

	// SPONZA MODEL //

	model = ModelLoaderInfo();
	model.modelFilePath = "data/models/dabrovic-sponza/sponza.obj";
	//model.scale = Vector3(0.01f, 0.01f, 0.01f);

	scene.models.push_back(model);

	// LIGHTS //

	AmbientLight ambientLight;
	ambientLight.color = skyColor * 0.05f;

	scene.lights.ambientLights.push_back(ambientLight);
	
	PointLight pointLight;
	pointLight.color = skyColor * 200.0f;
	pointLight.position = Vector3(0.0f, 18.0f, 0.0f);

	scene.lights.pointLights.push_back(pointLight);

	DirectionalLight directionalLight;
	directionalLight.color = skyColor * 1.0f;
	directionalLight.direction = EulerAngle(-80.0f, 0.0f, 0.0f).getDirection();

	//scene.lights.directionalLights.push_back(directionalLight);

	return scene;
}

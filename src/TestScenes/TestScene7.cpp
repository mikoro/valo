// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// BUDDHA + DRAGON //

Scene TestScene::create7()
{
	Scene scene;

	Color skyColor(182, 126, 91);

	scene.general.tracerType = TracerType::RAY;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::TENT;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(0.0f, 0.6f, 1.5f);
	scene.camera.orientation = EulerAngle(-8.0f, -0.0f, 0.0f);

	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	// MODELS //

	ModelLoaderInfo planeModel;
	planeModel.modelFilePath = "data/models/plane.obj";
	planeModel.defaultMaterialId = 1;
	planeModel.scale = Vector3(10.0f, 10.0f, 10.0f);

	scene.models.push_back(planeModel);

	ModelLoaderInfo buddhaModel;
	buddhaModel.modelFilePath = "data/models/buddha.obj";
	buddhaModel.defaultMaterialId = 2;
	buddhaModel.triangleCountEstimate = 1087451;
	buddhaModel.translate = Vector3(0.6f, 0.0f, 0.0f);
	buddhaModel.rotate = EulerAngle(0.0f, 150.0f, 0.0f);
	
	scene.models.push_back(buddhaModel);

	ModelLoaderInfo dragonModel;
	dragonModel.modelFilePath = "data/models/dragon.obj";
	dragonModel.defaultMaterialId = 3;
	dragonModel.triangleCountEstimate = 871306;
	dragonModel.translate = Vector3(-0.4f, 0.0f, 0.0f);
	dragonModel.rotate = EulerAngle(0.0f, 110.0f, 0.0f);

	scene.models.push_back(dragonModel);

	// MATERIALS //

	DiffuseSpecularMaterial planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.diffuseReflectance = Color::WHITE;
	planeMaterial.reflectance = Color::WHITE;

	scene.materials.diffuseSpecularMaterials.push_back(planeMaterial);

	DiffuseSpecularMaterial buddhaMaterial;
	buddhaMaterial.id = 2;
	buddhaMaterial.diffuseReflectance = Color::WHITE;
	buddhaMaterial.reflectance = Color::WHITE;

	scene.materials.diffuseSpecularMaterials.push_back(buddhaMaterial);

	DiffuseSpecularMaterial dragonMaterial;
	dragonMaterial.id = 3;
	dragonMaterial.diffuseReflectance = Color::WHITE;
	dragonMaterial.reflectance = Color::WHITE;

	scene.materials.diffuseSpecularMaterials.push_back(dragonMaterial);

	// LIGHTS //

	AmbientLight ambientLight;
	ambientLight.color = skyColor * 0.05f;

	scene.lights.ambientLights.push_back(ambientLight);
	
	PointLight pointLight;
	pointLight.color = skyColor * 2.0f;
	pointLight.position = Vector3(0.0f, 2.0f, 2.0f);

	scene.lights.pointLights.push_back(pointLight);

	return scene;
}

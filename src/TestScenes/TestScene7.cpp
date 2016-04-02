// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// BUDDHA + DRAGON //

Scene TestScene::create7()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;
	
	scene.camera.position = Vector3(0.0f, 0.6f, 1.5f);
	scene.camera.orientation = EulerAngle(-8.0f, -0.0f, 0.0f);

	// MODELS //

	ModelLoaderInfo planeModel;
	planeModel.modelFileName = "data/models/plane.obj";
	planeModel.defaultMaterialId = 1;
	planeModel.scale = Vector3(10.0f, 10.0f, 10.0f);

	scene.models.push_back(planeModel);

	ModelLoaderInfo buddhaModel;
	buddhaModel.modelFileName = "data/models/buddha.obj";
	buddhaModel.defaultMaterialId = 2;
	buddhaModel.triangleCountEstimate = 1087451;
	buddhaModel.translate = Vector3(0.6f, 0.0f, 0.0f);
	buddhaModel.rotate = EulerAngle(0.0f, 150.0f, 0.0f);
	
	scene.models.push_back(buddhaModel);

	ModelLoaderInfo dragonModel;
	dragonModel.modelFileName = "data/models/dragon.obj";
	dragonModel.defaultMaterialId = 3;
	dragonModel.triangleCountEstimate = 871306;
	dragonModel.translate = Vector3(-0.4f, 0.0f, 0.0f);
	dragonModel.rotate = EulerAngle(0.0f, 110.0f, 0.0f);

	scene.models.push_back(dragonModel);

	// MATERIALS //

	Material planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.reflectance = Color(1.0f, 1.0f, 1.0f);
	
	scene.materials.push_back(planeMaterial);

	Material buddhaMaterial;
	buddhaMaterial.id = 2;
	buddhaMaterial.reflectance = Color(1.0f, 1.0f, 1.0f);
	
	scene.materials.push_back(buddhaMaterial);

	Material dragonMaterial;
	dragonMaterial.id = 3;
	dragonMaterial.reflectance = Color(1.0f, 1.0f, 1.0f);
	
	scene.materials.push_back(dragonMaterial);

	return scene;
}

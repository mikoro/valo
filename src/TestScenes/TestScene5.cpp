// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// DABROVIC SPONZA //

Scene TestScene::create5()
{
	Scene scene;

	Color skyColor(182, 126, 91);

	scene.integrator.type = IntegratorType::DOT;

	scene.camera.position = Vector3(8.92f, 0.68f, -2.02f);
	scene.camera.orientation = EulerAngle(6.66f, 111.11f, 0.0f);

	scene.bvh.type = BVHType::BVH4;

	// PLANE MODEL //

	Material planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.reflectance = skyColor;
	planeMaterial.emittance = skyColor * 10.0f;
	planeMaterial.invertNormal = false;

	ModelLoaderInfo model;
	model.modelFileName = "data/models/plane.obj";
	model.defaultMaterialId = planeMaterial.id;
	model.scale = Vector3(11.0f, 1.0f, 3.0f);
	model.translate = Vector3(0.0f, 15.7f, 0.0f);
	model.rotate = EulerAngle(180.0f, 0.0f, 0.0f);

	scene.materials.push_back(planeMaterial);
	scene.models.push_back(model);

	// SPONZA MODEL //

	model = ModelLoaderInfo();
	model.modelFileName = "data/models/dabrovic-sponza/sponza.obj";
	//model.scale = Vector3(0.01f, 0.01f, 0.01f);

	scene.models.push_back(model);

	return scene;
}

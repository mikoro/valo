// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Tracing/Scene.h"

using namespace Raycer;

// CRYTEK SPONZA //

Scene TestScene::create6()
{
	Scene scene;

	Color skyColor(182, 126, 91);

	scene.general.tracerType = TracerType::PATH;
	scene.pathtracing.enableMultiSampling = true;
	scene.pathtracing.multiSamplerFilterType = FilterType::TENT;
	scene.pathtracing.minPathLength = 3;
	scene.pathtracing.terminationProbability = 0.5f;
	scene.pathtracing.pixelSampleCount = 1;

	scene.camera.position = Vector3(8.92f, 0.68f, -2.02f);
	scene.camera.orientation = EulerAngle(6.66f, 111.11f, 0.0f);

	scene.bvhInfo.bvhType = BVHType::BVH4;
	scene.bvhInfo.maxLeafSize = 4;

	scene.bvhInfo.loadFromFile = false;
	scene.imagePoolInfo.loadFromFile = false;

	// PLANE MODEL //

	DefaultMaterial planeMaterial;
	planeMaterial.id = 1;
	planeMaterial.reflectance = skyColor;
	planeMaterial.emittance = skyColor * 10.0f;
	planeMaterial.invertNormal = false;
	planeMaterial.skipLighting = true;
	planeMaterial.nonShadowing = true;

	ModelLoaderInfo model;
	model.modelFilePath = "data/models/plane.obj";
	model.defaultMaterialId = planeMaterial.id;
	model.scale = Vector3(9.0f, 1.0f, 2.0f);
	model.translate = Vector3(-0.6f, 13.0f, -0.3f);
	model.rotate = EulerAngle(180.0f, 0.0f, 0.0f);

	scene.materials.defaultMaterials.push_back(planeMaterial);
	scene.models.push_back(model);

	// SPONZA MODEL //

	model = ModelLoaderInfo();
	model.modelFilePath = "data/models/crytek-sponza/sponza.obj";
	model.scale = Vector3(0.01f, 0.01f, 0.01f);

	scene.models.push_back(model);

	return scene;
}

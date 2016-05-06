// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "TestScenes/TestScene.h"
#include "Core/Scene.h"

using namespace Raycer;

// CRYTEK SPONZA 1 //

Scene TestScene::create5()
{
	Scene scene;

	scene.integrator.type = IntegratorType::DOT;
	scene.integrator.aoIntegrator.maxDistance = 0.5f;
	scene.renderer.filter.type = FilterType::MITCHELL;
	scene.tonemapper.type = TonemapperType::REINHARD;
	scene.tonemapper.reinhardTonemapper.key = 0.08f;

	scene.camera.position = Vector3(-8.4359f, 1.1074f, -0.5245f);
	scene.camera.orientation = EulerAngle(-20.3729f, -90.0f, 0.0f);
	scene.camera.fov = 100.0f;

	scene.bvh.type = BVHType::BVH2;

	// MATERIALS //

	Texture texture1;
	texture1.type = TextureType::WOOD;
	texture1.id = 1;
	texture1.woodTexture.scale = 1.0f;
	texture1.woodTexture.density = 32.0f;
	texture1.woodTexture.bumpiness = 3.0f;

	Material material1;
	material1.type = MaterialType::DIFFUSE;
	material1.id = 1;
	material1.reflectanceTextureId = 1;

	scene.textures.push_back(texture1);
	scene.materials.push_back(material1);

	Texture texture2;
	texture2.type = TextureType::MARBLE;
	texture2.id = 2;
	texture2.marbleTexture.scale = 4.0f;
	
	Material material2;
	material2.type = MaterialType::DIFFUSE;
	material2.id = 2;
	material2.reflectanceTextureId = 2;

	scene.textures.push_back(texture2);
	scene.materials.push_back(material2);

	Texture texture3;
	texture3.type = TextureType::FIRE;
	texture3.id = 3;
	texture3.fireTexture.scale = 4.0f;

	Material material3;
	material3.type = MaterialType::DIFFUSE;
	material3.id = 3;
	material3.reflectanceTextureId = 3;

	scene.textures.push_back(texture3);
	scene.materials.push_back(material3);

	// SPONZA MODEL //

	ModelLoaderInfo model;
	model.modelFileName = "data/models/crytek-sponza/sponza.obj";
	model.scale = Vector3(0.01f, 0.01f, 0.01f);

	scene.models.push_back(model);

	return scene;
}

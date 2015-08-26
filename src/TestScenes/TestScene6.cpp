// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Raytracing/Scene.h"

using namespace Raycer;

// spherical environment texture map with glass primitives
Scene Scene::createTestScene6()
{
	Scene scene;

	scene.raytracer.maxRayIterations = 8;
	scene.raytracer.multiSamples = 0;

	scene.toneMapper.type = ToneMapperType::REINHARD;

	// CAMERA //

	scene.camera.position = Vector3(7.0, 0.0, -12.2);
	scene.camera.orientation = EulerAngle(0.0, 150.0, 0.0);

	// SPHERE 1 //

	ImageTexture sphere1Texture;
	sphere1Texture.id = 1;
	sphere1Texture.intensity = 1.0;
	sphere1Texture.imageFilePath = "data/images/rooftop.hdr";
	sphere1Texture.applyGamma = false;

	Material sphere1Material;
	sphere1Material.id = 1;
	sphere1Material.colorTextureId = sphere1Texture.id;
	sphere1Material.skipLighting = true;
	sphere1Material.texcoordScale = Vector2(1.0, -1.0);

	Sphere sphere1;
	sphere1.materialId = sphere1Material.id;
	sphere1.position = Vector3(0.0, 0.0, 0.0);
	sphere1.radius = 100.0;

	scene.textures.imageTextures.push_back(sphere1Texture);
	scene.materials.push_back(sphere1Material);
	scene.primitives.spheres.push_back(sphere1);

	// SPHERE 2 //

	ColorTexture sphere2Texture;
	sphere2Texture.id = 2;
	sphere2Texture.color = Color(1.0, 1.0, 1.0);
	sphere2Texture.intensity = 1.0;

	Material sphere2Material;
	sphere2Material.id = 2;
	sphere2Material.colorTextureId = sphere2Texture.id;
	sphere2Material.rayReflectance = 1.0;
	sphere2Material.rayTransmittance = 1.0;
	sphere2Material.refractiveIndex = 1.5;
	sphere2Material.isFresnel = true;
	sphere2Material.enableAttenuation = false;
	sphere2Material.attenuation = 0.1;
	sphere2Material.attenuationColor = Color(0.0, 0.0, 0.1);

	Sphere sphere2;
	sphere2.materialId = sphere2Material.id;
	sphere2.position = Vector3(4.0, -2.0, 0.0);
	sphere2.radius = 2.0;

	scene.textures.colorTextures.push_back(sphere2Texture);
	scene.materials.push_back(sphere2Material);
	scene.primitives.spheres.push_back(sphere2);

	// SPHERE 3 //

	ColorTexture sphere3Texture;
	sphere3Texture.id = 3;
	sphere3Texture.color = Color(1.0, 1.0, 1.0);
	sphere3Texture.intensity = 1.0;

	Material sphere3Material;
	sphere3Material.id = 3;
	sphere3Material.colorTextureId = sphere3Texture.id;
	sphere3Material.rayReflectance = 1.0;
	sphere3Material.rayTransmittance = 1.0;
	sphere3Material.refractiveIndex = 1.5;
	sphere3Material.isFresnel = true;
	sphere3Material.enableAttenuation = false;
	sphere3Material.attenuation = 0.1;
	sphere3Material.attenuationColor = Color(0.1, 0.0, 0.0);

	Sphere sphere3;
	sphere3.materialId = sphere3Material.id;
	sphere3.position = Vector3(-4.0, 2.0, 0.0);
	sphere3.radius = 2.0;

	scene.textures.colorTextures.push_back(sphere3Texture);
	scene.materials.push_back(sphere3Material);
	scene.primitives.spheres.push_back(sphere3);

	// BOX 1 //

	ColorTexture box1Texture;
	box1Texture.id = 4;
	box1Texture.color = Color(1.0, 1.0, 1.0);
	box1Texture.intensity = 1.0;

	Material box1Material;
	box1Material.id = 4;
	box1Material.colorTextureId = box1Texture.id;
	box1Material.rayReflectance = 1.0;
	box1Material.rayTransmittance = 1.0;
	box1Material.refractiveIndex = 1.5;
	box1Material.isFresnel = true;
	box1Material.enableAttenuation = true;
	box1Material.attenuation = 0.4;
	box1Material.attenuationColor = Color(0.0, 0.0, 0.0);

	Box box1;
	box1.position = Vector3(0.0, 0.0, 0.0);
	box1.extent = Vector3(1.0, 10.0, 6.0);
	box1.materialId = box1Material.id;

	scene.textures.colorTextures.push_back(box1Texture);
	scene.materials.push_back(box1Material);
	scene.primitives.boxes.push_back(box1);

	return scene;
}

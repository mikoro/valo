// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"
#include "Materials/Material.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Raycer;

CUDA_CALLABLE Vector3 Material::getDirection(const Intersection& intersection, Random& random)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getDirection(*this, intersection, random);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getDirection(*this, intersection, random);
		default: return Vector3();
	}
}

CUDA_CALLABLE Color Material::getBrdf(const Scene& scene, const Intersection& intersection, const Vector3& in, const Vector3& out)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getBrdf(scene, *this, intersection, in, out);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getBrdf(scene, *this, intersection, in, out);
		default: return Color::black();
	}
}

CUDA_CALLABLE float Material::getPdf(const Intersection& intersection, const Vector3& out)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getPdf(*this, intersection, out);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getPdf(*this, intersection, out);
		default: return 0.0f;
	}
}

CUDA_CALLABLE bool Material::isEmissive() const
{
	return emittanceTextureIndex != -1 || !emittance.isZero();
}

CUDA_CALLABLE Color Material::getEmittance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
	if (emittanceTextureIndex != -1)
		return scene.getTexture(emittanceTextureIndex).getColor(scene, texcoord, position);
	else
		return emittance;
}

CUDA_CALLABLE Color Material::getReflectance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
	if (reflectanceTextureIndex != -1)
		return scene.getTexture(reflectanceTextureIndex).getColor(scene, texcoord, position);
	else
		return reflectance;
}

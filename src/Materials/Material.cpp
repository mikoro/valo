// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Materials/Material.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Raycer;

Vector3 Material::getDirection(const Intersection& intersection, Random& random)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getDirection(*this, intersection, random);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getDirection(*this, intersection, random);
		default: return Vector3();
	}
}

Color Material::getBrdf(const Intersection& intersection, const Vector3& in, const Vector3& out)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getBrdf(*this, intersection, in, out);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getBrdf(*this, intersection, in, out);
		default: return Color::BLACK;
	}
}

float Material::getPdf(const Intersection& intersection, const Vector3& out)
{
	switch (type)
	{
		case MaterialType::DIFFUSE: return diffuseMaterial.getPdf(*this, intersection, out);
		case MaterialType::BLINN_PHONG: return blinnPhongMaterial.getPdf(*this, intersection, out);
		default: return 0.0f;
	}
}

bool Material::isEmissive() const
{
	return emittanceTexture != nullptr || !emittance.isZero();
}

Color Material::getEmittance(const Vector2& texcoord, const Vector3& position) const
{
	if (emittanceTexture != nullptr)
		return emittanceTexture->getColor(texcoord, position);
	else
		return emittance;
}

Color Material::getReflectance(const Vector2& texcoord, const Vector3& position) const
{
	if (reflectanceTexture != nullptr)
		return reflectanceTexture->getColor(texcoord, position);
	else
		return reflectance;
}

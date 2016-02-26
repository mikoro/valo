// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/Material.h"
#include "Tracing/Intersection.h"
#include "Textures/Texture.h"

using namespace Raycer;

Color Material::getReflectance(const Intersection& intersection)
{
	if (reflectanceMapTexture != nullptr)
		return reflectanceMapTexture->getColor(intersection.texcoord, intersection.position);
	else
		return reflectance;
}

Color Material::getEmittance(const Intersection& intersection)
{
	if (emittanceMapTexture != nullptr)
		return emittanceMapTexture->getColor(intersection.texcoord, intersection.position);
	else
		return emittance;
}

Color Material::getAmbientReflectance(const Intersection& intersection)
{
	if (ambientMapTexture != nullptr)
		return ambientMapTexture->getColor(intersection.texcoord, intersection.position);
	else
		return ambientReflectance;
}

Color Material::getDiffuseReflectance(const Intersection& intersection)
{
	if (diffuseMapTexture != nullptr)
		return diffuseMapTexture->getColor(intersection.texcoord, intersection.position);
	else
		return diffuseReflectance;
}

Color Material::getSpecularReflectance(const Intersection& intersection)
{
	if (specularMapTexture != nullptr)
		return specularMapTexture->getColor(intersection.texcoord, intersection.position);
	else
		return specularReflectance;
}

bool Material::isEmissive()
{
	if (emittanceMapTexture != nullptr)
		return true;
	else if (!emittance.isZero())
		return true;
	else
		return false;
}

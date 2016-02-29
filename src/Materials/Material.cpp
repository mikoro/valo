// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/Material.h"
#include "Tracing/Intersection.h"
#include "Textures/Texture.h"

using namespace Raycer;

Color Material::getReflectance(const Intersection& intersection)
{
	if (reflectanceTexture != nullptr)
		return reflectanceTexture->getColor(intersection.texcoord, intersection.position);
	else
		return reflectance;
}

Color Material::getEmittance(const Intersection& intersection)
{
	if (emittanceTexture != nullptr)
		return emittanceTexture->getColor(intersection.texcoord, intersection.position);
	else
		return emittance;
}

bool Material::isEmissive()
{
	return (emittanceTexture != nullptr) || (!emittance.isZero());
}

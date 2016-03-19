// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Materials/DiffuseMaterial.h"
#include "Materials/Material.h"
#include "Math/Mapper.h"
#include "Utils/Random.h"

using namespace Raycer;

Vector3 DiffuseMaterial::getDirection(const Material& material, const Intersection& intersection, Random& random)
{
	(void)material;

	return Mapper::mapToCosineHemisphere(random.getVector2(), intersection.onb);
}

Color DiffuseMaterial::getBrdf(const Material& material, const Intersection& intersection, const Vector3& out)
{
	(void)out;

	return material.getReflectance(intersection.texcoord, intersection.position) / float(M_PI);
}

float DiffuseMaterial::getPdf(const Material& material, const Intersection& intersection, const Vector3& out)
{
	(void)material;

	return intersection.normal.dot(out) / float(M_PI);
}

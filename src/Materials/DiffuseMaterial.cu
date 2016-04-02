// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Materials/DiffuseMaterial.h"
#include "Materials/Material.h"
#include "Math/Mapper.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Vector3 DiffuseMaterial::getDirection(const Material& material, const Intersection& intersection, Random& random)
{
	(void)material;

	return Mapper::mapToCosineHemisphere(random.getVector2(), intersection.onb);
}

CUDA_CALLABLE Color DiffuseMaterial::getBrdf(const Scene& scene, const Material& material, const Intersection& intersection, const Vector3& in, const Vector3& out)
{
	(void)in;
	(void)out;

	return material.getReflectance(scene, intersection.texcoord, intersection.position) / float(M_PI);
}

CUDA_CALLABLE float DiffuseMaterial::getPdf(const Material& material, const Intersection& intersection, const Vector3& out)
{
	(void)material;

	return intersection.normal.dot(out) / float(M_PI);
}

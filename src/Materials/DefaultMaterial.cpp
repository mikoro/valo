// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DefaultMaterial.h"
#include "Tracing/Scene.h"
#include "Tracing/Intersection.h"
#include "Samplers/RandomSampler.h"

using namespace Raycer;

Vector3 DefaultMaterial::getDirection(const Intersection& intersection, RandomSampler& sampler, Random& random)
{
	return sampler.getCosineHemisphereSample(intersection.onb, 0, 0, 0, 0, 0, random);
}

float DefaultMaterial::getProbability(const Intersection& intersection, const Vector3& out)
{
	return intersection.normal.dot(out) / float(M_PI);
}

Color DefaultMaterial::getBrdf(const Intersection& intersection, const Vector3& out)
{
	(void)out;

	return getReflectance(intersection) / float(M_PI);
}

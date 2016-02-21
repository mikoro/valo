// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseMaterial.h"
#include "Tracing/Scene.h"
#include "Tracing/Intersection.h"
#include "Samplers/RandomSampler.h"

using namespace Raycer;

Color DiffuseMaterial::getColor(const Scene& scene, const Intersection& intersection, const Light& light, Random& random)
{
	Color lightColor = light.getColor(scene, intersection, random);
	Color finalColor = lightColor * getAmbientReflectance(intersection);

	if (light.hasDirection())
	{
		Vector3 directionToLight = -light.getDirection(intersection);
		
		float diffuseAmount = directionToLight.dot(intersection.normal);

		if (diffuseAmount > 0.0f)
			finalColor += lightColor * diffuseAmount * getDiffuseReflectance(intersection);
	}

	return finalColor;
}

Vector3 DiffuseMaterial::getSampleDirection(const Intersection& intersection, RandomSampler& sampler, Random& random)
{
	return sampler.getCosineHemisphereSample(intersection.onb, 0, 0, 0, 0, 0, random);
}

float DiffuseMaterial::getDirectionProbability(const Intersection& intersection, const Vector3& out)
{
	return intersection.normal.dot(out) / float(M_PI);
}

Color DiffuseMaterial::getBrdf(const Intersection& intersection, const Vector3& out)
{
	(void)out;

	return getReflectance(intersection) / float(M_PI);
}

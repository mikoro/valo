// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseMaterial.h"
#include "Scenes/Scene.h"
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
		
		double diffuseAmount = directionToLight.dot(intersection.normal);

		if (diffuseAmount > 0.0)
			finalColor += lightColor * diffuseAmount * getDiffuseReflectance(intersection);
	}

	return finalColor;
}

void DiffuseMaterial::getSample(const Intersection& intersection, RandomSampler& sampler, Random& random, Vector3& newDirection, double& pdf)
{
	newDirection = sampler.getUniformHemisphereSample(intersection.onb, 0, 0, 0, 0, 0, random);
	pdf = intersection.normal.dot(newDirection) / M_PI;
}

double DiffuseMaterial::getBrdf(const Intersection& intersection, const Vector3& newDirection)
{
	(void)intersection;
	(void)newDirection;

	return 1.0;
}

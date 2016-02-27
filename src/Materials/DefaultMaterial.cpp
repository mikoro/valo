// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DefaultMaterial.h"
#include "Tracing/Scene.h"
#include "Tracing/Intersection.h"
#include "Samplers/RandomSampler.h"

using namespace Raycer;

Color DefaultMaterial::getColor(const Scene& scene, const Intersection& intersection, const Light& light, Random& random)
{
	Color intersectionReflectance = getReflectance(intersection);
	Color lightColor = light.getColor(scene, intersection, random);
	Color finalColor = lightColor * intersectionReflectance;

	if (light.hasDirection())
	{
		Vector3 directionToLight = -light.getDirection(intersection);
		Vector3 directionToCamera = -intersection.rayDirection;

		float diffuseAmount = directionToLight.dot(intersection.normal);

		if (diffuseAmount > 0.0f)
		{
			finalColor += lightColor * diffuseAmount * intersectionReflectance;

			Vector3 reflectionDirection = ((2.0f * diffuseAmount * intersection.normal) - directionToLight).normalized();
			float specularAmount = reflectionDirection.dot(directionToCamera);

			if (specularAmount > 0.0f)
				finalColor += lightColor * std::pow(specularAmount, shininess) * getSpecularReflectance(intersection);
		}
	}

	return finalColor;
}

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

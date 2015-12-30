// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseSpecularMaterial.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"
#include "Samplers/RandomSampler.h"

using namespace Raycer;

Color DiffuseSpecularMaterial::getColor(const Scene& scene, const Intersection& intersection, const Light& light, Random& random)
{
	Color lightColor = light.getColor(scene, intersection, random);
	Color finalColor = lightColor * getAmbientReflectance(intersection);

	if (light.hasDirection())
	{
		Vector3 directionToLight = -light.getDirection(intersection);
		Vector3 directionToCamera = -intersection.rayDirection;

		double diffuseAmount = directionToLight.dot(intersection.normal);

		if (diffuseAmount > 0.0)
		{
			finalColor += lightColor * diffuseAmount * getDiffuseReflectance(intersection);

			Vector3 reflectionDirection = ((2.0 * diffuseAmount * intersection.normal) - directionToLight).normalized();
			double specularAmount = reflectionDirection.dot(directionToCamera);

			if (specularAmount > 0.0)
				finalColor += lightColor * pow(specularAmount, shininess) * getSpecularReflectance(intersection);
		}
	}

	return finalColor;
}

void DiffuseSpecularMaterial::getSample(const Intersection& intersection, RandomSampler& sampler, Random& random, Vector3& newDirection, double& pdf)
{
	newDirection = sampler.getCosineHemisphereSample(intersection.onb, 0, 0, 0, 0, 0, random);
	pdf = intersection.normal.dot(newDirection) / M_PI;
}

Color DiffuseSpecularMaterial::getBrdf(const Intersection& intersection, const Vector3& newDirection)
{
	(void)newDirection;

	return getReflectance(intersection) / M_PI;
}

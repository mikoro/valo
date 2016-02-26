// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseSpecularMaterial.h"
#include "Tracing/Scene.h"
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

		float diffuseAmount = directionToLight.dot(intersection.normal);

		if (diffuseAmount > 0.0f)
		{
			finalColor += lightColor * diffuseAmount * getDiffuseReflectance(intersection);

			Vector3 reflectionDirection = ((2.0f * diffuseAmount * intersection.normal) - directionToLight).normalized();
			float specularAmount = reflectionDirection.dot(directionToCamera);

			if (specularAmount > 0.0f)
				finalColor += lightColor * std::pow(specularAmount, shininess) * getSpecularReflectance(intersection);
		}
	}

	return finalColor;
}

Vector3 DiffuseSpecularMaterial::getSampleDirection(const Intersection& intersection, RandomSampler& sampler, Random& random)
{
	return sampler.getCosineHemisphereSample(intersection.onb, 0, 0, 0, 0, 0, random);
}

float DiffuseSpecularMaterial::getDirectionProbability(const Intersection& intersection, const Vector3& out)
{
	return intersection.normal.dot(out) / float(M_PI);
}

Color DiffuseSpecularMaterial::getBrdf(const Intersection& intersection, const Vector3& out)
{
	(void)out;

	return getReflectance(intersection) / float(M_PI);
}

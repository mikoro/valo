// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseSpecularMaterial.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"

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

Vector3 DiffuseSpecularMaterial::getDirection(const Intersection& intersection, Random& random)
{
	(void)intersection;
	(void)random;

	return Vector3();
}

double DiffuseSpecularMaterial::getBrdf(const Vector3& in, const Vector3& normal, const Vector3& out)
{
	(void)in;
	(void)normal;
	(void)out;

	return 0.0;
}

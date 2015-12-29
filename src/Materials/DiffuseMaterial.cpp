// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Materials/DiffuseMaterial.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"

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

Vector3 DiffuseMaterial::getDirection(const Intersection& intersection, Random& random)
{
	(void)intersection;
	(void)random;

	return Vector3();
}

double DiffuseMaterial::getBrdf(const Vector3& in, const Vector3& normal, const Vector3& out)
{
	(void)in;
	(void)normal;
	(void)out;

	return 0.0;
}

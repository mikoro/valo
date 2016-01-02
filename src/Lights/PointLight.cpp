// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Lights/PointLight.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"
#include "Tracing/Ray.h"

using namespace Raycer;

void PointLight::initialize()
{
}

bool PointLight::hasDirection() const
{
	return true;
}

Color PointLight::getColor(const Scene& scene, const Intersection& intersection, Random& random) const
{
	(void)random;

	Vector3 intersectionToLight = position - intersection.position;
	double distance2 = intersectionToLight.lengthSquared();
	double distance = std::sqrt(distance2);
	Vector3 directionToLight = intersectionToLight / distance;
	double cosine = intersection.normal.dot(directionToLight);

	if (cosine < 0.0)
		return Color(0.0, 0.0, 0.0);

	Ray shadowRay;
	Intersection shadowIntersection;

	shadowRay.origin = intersection.position;
	shadowRay.direction = directionToLight;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = distance;
	shadowRay.precalculate();

	if (scene.intersect(shadowRay, shadowIntersection))
		return Color(0.0, 0.0, 0.0);

	return cosine * color / distance2;
}

Vector3 PointLight::getDirection(const Intersection& intersection) const
{
	return (intersection.position - position).normalized();
}

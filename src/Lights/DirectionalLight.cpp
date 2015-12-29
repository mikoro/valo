// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Lights/DirectionalLight.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"
#include "Tracing/Ray.h"

using namespace Raycer;

void DirectionalLight::initialize()
{
}

bool DirectionalLight::hasDirection() const
{
	return true;
}

Color DirectionalLight::getColor(const Scene& scene, const Intersection& intersection, Random& random) const
{
	(void)random;

	if (intersection.normal.dot(direction) > 0.0)
		return Color(0.0, 0.0, 0.0);

	Ray shadowRay;
	Intersection shadowIntersection;

	shadowRay.origin = intersection.position;
	shadowRay.direction = -direction;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = std::numeric_limits<double>::max();
	shadowRay.precalculate();

	if (scene.intersect(shadowRay, shadowIntersection))
		return Color(0.0, 0.0, 0.0);

	return color;
}

Vector3 DirectionalLight::getDirection(const Intersection& intersection) const
{
	(void)intersection;

	return direction;
}

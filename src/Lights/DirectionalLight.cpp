// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Lights/DirectionalLight.h"
#include "Tracing/Scene.h"
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

	float cosine = intersection.normal.dot(-direction);

	if (cosine < 0.0f)
		return Color(0.0f, 0.0f, 0.0f);

	Ray shadowRay;
	Intersection shadowIntersection;

	shadowRay.origin = intersection.position;
	shadowRay.direction = -direction;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = std::numeric_limits<float>::max();
	shadowRay.precalculate();

	if (scene.intersect(shadowRay, shadowIntersection))
		return Color(0.0f, 0.0f, 0.0f);

	return cosine * color;
}

Vector3 DirectionalLight::getDirection(const Intersection& intersection) const
{
	(void)intersection;

	return direction;
}

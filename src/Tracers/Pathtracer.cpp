// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Pathtracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

using namespace Raycer;

Color Pathtracer::trace(const Scene& scene, const Ray& ray, Random& random)
{
	return traceRecursive(scene, ray, random);
}

Color Pathtracer::traceRecursive(const Scene& scene, const Ray& ray, Random& random)
{
	Intersection intersection;
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return Color(0.0, 0.0, 0.0);

	Material* material = intersection.material;

	Color emittance = material->getEmittance(intersection);

	if (random.getDouble() < scene.pathtracing.terminationProbability)
		return emittance;

	Vector3 newDirection;
	double pdf;

	material->getSample(intersection, sampler, random, newDirection, pdf);

	Color brdf = material->getBrdf(intersection, newDirection);
	double cosine = newDirection.dot(intersection.normal);

	Ray newRay;
	newRay.origin = intersection.position;
	newRay.direction = newDirection;
	newRay.minDistance = scene.general.rayMinDistance;
	newRay.precalculate();

	return emittance + brdf * cosine * traceRecursive(scene, newRay, random) / pdf / scene.pathtracing.terminationProbability;
}

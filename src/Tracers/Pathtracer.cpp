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
	return traceRecursive(scene, ray, 0, random);
}

Color Pathtracer::traceRecursive(const Scene& scene, const Ray& ray, uint64_t iteration, Random& random)
{
	if (iteration >= scene.pathtracing.maxPathLength)
		return Color::BLACK;

	Intersection intersection;
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return Color::BLACK;

	Material* material = intersection.material;

	if (material->emissive)
	{
		Color emittance = material->emittance;

		if (material->emittanceMapTexture != nullptr)
			emittance = material->emittanceMapTexture->getColor(intersection.texcoord, intersection.position) * material->emittanceMapTexture->intensity;

		return emittance;
	}

	Sampler* sampler = samplers[SamplerType::RANDOM].get();
	Vector3 newDirection = sampler->getHemisphereSample(intersection.onb, 1.0, 0, 0, 1, 1, 0, random);

	Ray newRay;
	newRay.origin = intersection.position;
	newRay.direction = newDirection;
	newRay.minDistance = scene.general.rayMinDistance;
	newRay.precalculate();

	Color reflectance = material->diffuseReflectance;

	if (material->diffuseMapTexture != nullptr)
		reflectance = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position) * material->diffuseMapTexture->intensity;

	double alpha = std::abs(newDirection.dot(intersection.normal));
	Color brdf = 2.0 * reflectance * alpha;
	Color reflected = traceRecursive(scene, newRay, iteration + 1, random);

	return brdf * reflected;
}

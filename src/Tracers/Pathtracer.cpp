// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Pathtracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Rendering/Film.h"

using namespace Raycer;

uint64_t Pathtracer::getPixelSampleCount(const Scene& scene) const
{
	return scene.pathtracing.pixelSampleCount;
}

uint64_t Pathtracer::getSamplesPerPixel(const Scene& scene) const
{
	(void)scene;

	return 1;
}

void Pathtracer::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random)
{
	Vector2 offsetPixel = pixelCenter;
	double filterWeight = 1.0;

	if (scene.pathtracing.enableMultiSampling)
	{
		Filter* filter = filters[scene.pathtracing.multiSamplerFilterType].get();

		Vector2 pixelOffset = sampler.getSquareSample(0, 0, 0, 0, 0, random);
		pixelOffset = (pixelOffset - Vector2(0.5, 0.5)) * 2.0 * filter->getRadius();

		filterWeight = filter->getWeight(pixelOffset);
		offsetPixel = pixelCenter + pixelOffset;
	}

	bool isOffLens;
	Ray ray = scene.camera.getRay(offsetPixel, isOffLens);

	if (isOffLens)
	{
		film.addSample(pixelIndex, scene.general.offLensColor, filterWeight);
		return;
	}

	Color color = traceRecursive(scene, ray, random, 0);
	film.addSample(pixelIndex, color, filterWeight);
}

Color Pathtracer::traceRecursive(const Scene& scene, const Ray& ray, Random& random, uint64_t iteration)
{
	Intersection intersection;
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return Color(0.0, 0.0, 0.0);

	Material* material = intersection.material;

	Color emittance = material->getEmittance(intersection);
	double terminationProbability = 1.0;

	if (iteration >= scene.pathtracing.minPathLength)
	{
		terminationProbability = scene.pathtracing.terminationProbability;

		if (random.getDouble() < terminationProbability)
			return emittance;
	}
	
	Vector3 reflectionDirection = material->getNewDirection(intersection, sampler, random);
	Color reflectionBrdf = material->getBrdf(intersection, reflectionDirection);
	double reflectionPdf = material->getPdf(intersection, reflectionDirection);
	double reflectionCosine = reflectionDirection.dot(intersection.normal);

	Ray reflectedRay;
	reflectedRay.origin = intersection.position;
	reflectedRay.direction = reflectionDirection;
	reflectedRay.minDistance = scene.general.rayMinDistance;
	reflectedRay.precalculate();

	return emittance + reflectionBrdf * reflectionCosine * traceRecursive(scene, reflectedRay, random, iteration + 1) / reflectionPdf / terminationProbability;
}

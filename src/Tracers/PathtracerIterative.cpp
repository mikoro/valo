// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PathtracerIterative.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Rendering/Film.h"

using namespace Raycer;

uint64_t PathtracerIterative::getPixelSampleCount(const Scene& scene) const
{
	return scene.pathtracing.pixelSampleCount;
}

uint64_t PathtracerIterative::getSamplesPerPixel(const Scene& scene) const
{
	(void)scene;

	return 1;
}

void PathtracerIterative::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& pathCount)
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

	Color color(0.0, 0.0, 0.0);
	Color sampleBrdf(1.0, 1.0, 1.0);
	double sampleProbability = 1.0;
	uint64_t depth = 0;

	while (true)
	{
		++pathCount;

		Intersection intersection;
		scene.intersect(ray, intersection);

		if (!intersection.wasFound)
			break;

		if (scene.general.enableNormalMapping && intersection.material->normalMapTexture != nullptr)
			calculateNormalMapping(intersection);

		Color emittedLight(0.0, 0.0, 0.0);
		Color directLight(0.0, 0.0, 0.0);

		if (depth == 0 && !intersection.isBehind && intersection.material->isEmissive())
			emittedLight = intersection.material->getEmittance(intersection);

		directLight = calculateDirectLight(scene, intersection, random);

		double terminationProbability = 1.0;

		if (depth >= scene.pathtracing.minPathLength)
		{
			terminationProbability = scene.pathtracing.terminationProbability;

			if (random.getDouble() < terminationProbability)
				break;
		}

		color += sampleBrdf * (emittedLight + directLight) / sampleProbability / terminationProbability;

		Vector3 sampleDirection = intersection.material->getSampleDirection(intersection, sampler, random);
		sampleProbability *= intersection.material->getDirectionProbability(intersection, sampleDirection);
		double sampleCosine = sampleDirection.dot(intersection.normal);
		sampleBrdf *= intersection.material->getBrdf(intersection, sampleDirection) * sampleCosine;

		ray = Ray();
		ray.origin = intersection.position;
		ray.direction = sampleDirection;
		ray.minDistance = scene.general.rayMinDistance;
		ray.precalculate();

		++depth;
	}

	film.addSample(pixelIndex, color, filterWeight);
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PathtracerIterative.h"
#include "Tracing/Scene.h"
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

void PathtracerIterative::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount)
{
	rayCount = 0;

	Vector2 offsetPixel = pixelCenter;
	float filterWeight = 1.0f;

	if (scene.pathtracing.enableMultiSampling)
	{
		Filter* filter = filters[scene.pathtracing.multiSamplerFilterType].get();

		Vector2 pixelOffset = sampler.getSquareSample(0, 0, 0, 0, 0, random);
		pixelOffset = (pixelOffset - Vector2(0.5f, 0.5f)) * 2.0f * filter->getRadius();

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

	Color color(0.0f, 0.0f, 0.0f);
	Color sampleBrdf(1.0f, 1.0f, 1.0f);
	float sampleCosine = 1.0f;
	float sampleProbability = 1.0f;
	float continuationProbability = 1.0f;
	uint64_t depth = 0;

	while (true)
	{
		pathCount++;

		Intersection intersection;
		scene.intersect(ray, intersection);

		if (!intersection.wasFound)
			break;

		if (scene.general.normalMapping && intersection.material->normalTexture != nullptr)
			calculateNormalMapping(intersection);

		if (depth == 0 && !intersection.isBehind && intersection.material->isEmissive())
		{
			color = intersection.material->getEmittance(intersection);
			break;
		}

		Color directLight = calculateDirectLight(scene, intersection, random);

		if (depth++ >= scene.pathtracing.minPathLength)
		{
			if (random.getFloat() < scene.pathtracing.terminationProbability)
				break;

			continuationProbability *= (1.0f - scene.pathtracing.terminationProbability);
		}

		color += sampleBrdf * sampleCosine * directLight / sampleProbability / continuationProbability;

		Vector3 sampleDirection = intersection.material->getDirection(intersection, sampler, random);
		sampleBrdf *= intersection.material->getBrdf(intersection, sampleDirection);
		sampleCosine *= sampleDirection.dot(intersection.normal);
		sampleProbability *= intersection.material->getProbability(intersection, sampleDirection);
		
		if (sampleProbability == 0.0f)
			break;

		ray = Ray();
		ray.origin = intersection.position;
		ray.direction = sampleDirection;
		ray.minDistance = scene.general.rayMinDistance;
		ray.precalculate();
	}

	film.addSample(pixelIndex, color, filterWeight);
}

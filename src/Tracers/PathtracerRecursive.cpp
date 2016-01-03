// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PathtracerRecursive.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Rendering/Film.h"

using namespace Raycer;

uint64_t PathtracerRecursive::getPixelSampleCount(const Scene& scene) const
{
	return scene.pathtracing.pixelSampleCount;
}

uint64_t PathtracerRecursive::getSamplesPerPixel(const Scene& scene) const
{
	(void)scene;

	return 1;
}

void PathtracerRecursive::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& pathCount)
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

	Color color = traceRecursive(scene, ray, random, 0, pathCount);
	film.addSample(pixelIndex, color, filterWeight);
}

Color PathtracerRecursive::traceRecursive(const Scene& scene, const Ray& ray, Random& random, uint64_t depth, uint64_t& pathCount)
{
	++pathCount;

	Intersection intersection;
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return Color(0.0, 0.0, 0.0);

	Color emittedLight(0.0, 0.0, 0.0);
	Color directLight(0.0, 0.0, 0.0);
	Color indirectLight(0.0, 0.0, 0.0);

	if (scene.general.enableNormalMapping && intersection.material->normalMapTexture != nullptr)
		calculateNormalMapping(intersection);

	if (depth == 0 && !intersection.isBehind && intersection.material->isEmissive())
		emittedLight = intersection.material->getEmittance(intersection);
	
	directLight = calculateDirectLight(scene, intersection, random);
	indirectLight = calculateIndirectLight(scene, intersection, random, depth, pathCount);

	return emittedLight + directLight + indirectLight;
}

Color PathtracerRecursive::calculateIndirectLight(const Scene& scene, const Intersection& intersection, Random& random, uint64_t depth, uint64_t& pathCount)
{
	double terminationProbability = 1.0;

	if (depth >= scene.pathtracing.minPathLength)
	{
		terminationProbability = scene.pathtracing.terminationProbability;

		if (random.getDouble() < terminationProbability)
			return Color(0.0, 0.0, 0.0);
	}

	Vector3 sampleDirection = intersection.material->getSampleDirection(intersection, sampler, random);
	double sampleProbability = intersection.material->getDirectionProbability(intersection, sampleDirection);
	Color sampleBrdf = intersection.material->getBrdf(intersection, sampleDirection);
	double sampleCosine = sampleDirection.dot(intersection.normal);

	Ray sampleRay;
	sampleRay.origin = intersection.position;
	sampleRay.direction = sampleDirection;
	sampleRay.minDistance = scene.general.rayMinDistance;
	sampleRay.precalculate();

	return sampleBrdf * sampleCosine * traceRecursive(scene, sampleRay, random, depth + 1, pathCount) / sampleProbability / terminationProbability;
}

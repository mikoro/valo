// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Pathtracer.h"
#include "Tracers/TracerCommon.h"
#include "Tracing/Scene.h"
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

void Pathtracer::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount)
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
			TracerCommon::calculateNormalMapping(intersection);

		if (depth == 0 && !intersection.isBehind && intersection.material->isEmissive())
		{
			color = intersection.material->getEmittance(intersection);
			break;
		}

		if (depth++ >= scene.pathtracing.minPathLength)
		{
			if (random.getFloat() < scene.pathtracing.terminationProbability)
				break;

			continuationProbability *= (1.0f - scene.pathtracing.terminationProbability);
		}

		Color directLight = calculateDirectLight(scene, intersection, random);
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

Color Pathtracer::calculateDirectLight(const Scene& scene, const Intersection& intersection, Random& random)
{
	uint64_t emitterCount = scene.emissiveTriangles.size();

	if (emitterCount == 0)
		return Color(0.0f, 0.0f, 0.0f);

	Triangle* emitter = scene.emissiveTriangles[random.getUint64(0, emitterCount - 1)];
	Intersection emitterIntersection = emitter->getRandomIntersection(random);
	Vector3 intersectionToEmitter = emitterIntersection.position - intersection.position;
	float emitterDistance2 = intersectionToEmitter.lengthSquared();
	float emitterDistance = sqrt(emitterDistance2);
	Vector3 sampleDirection = intersectionToEmitter / emitterDistance;

	Ray shadowRay;
	shadowRay.origin = intersection.position;
	shadowRay.direction = sampleDirection;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = emitterDistance - scene.general.rayMinDistance;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.precalculate();

	Intersection shadowIntersection;
	scene.intersect(shadowRay, shadowIntersection);

	if (shadowIntersection.wasFound)
		return Color(0.0f, 0.0f, 0.0f);

	float cosine1 = intersection.normal.dot(sampleDirection);
	float cosine2 = sampleDirection.dot(-emitter->normal);

	if (cosine1 < 0.0f || cosine2 < 0.0f)
		return Color(0.0f, 0.0f, 0.0f);

	float probability1 = 1.0f / float(emitterCount);
	float probability2 = 1.0f / emitter->getArea();

	Color emittance = emitter->material->getEmittance(emitterIntersection);
	Color intersectionBrdf = intersection.material->getBrdf(intersection, sampleDirection);

	return emittance * intersectionBrdf * cosine1 * cosine2 * (1.0f / emitterDistance2) / (probability1 * probability2);
}

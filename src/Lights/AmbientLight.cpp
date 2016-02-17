// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Lights/AmbientLight.h"
#include "Scenes/Scene.h"
#include "Tracing/Intersection.h"
#include "Tracing/Ray.h"

using namespace Raycer;

void AmbientLight::initialize()
{
	sampler = Sampler::getSampler(samplerType);
}

bool AmbientLight::hasDirection() const
{
	return false;
}

Color AmbientLight::getColor(const Scene& scene, const Intersection& intersection, Random& random) const
{
	if (!occlusion)
		return color;
	
	uint64_t occlusionCount = 0;
	uint64_t permutation = random.getUint64();
	uint64_t n = sampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector3 sampleDirection = sampler->getUniformHemisphereSample(intersection.onb, x, y, n, n, permutation, random);

			Ray sampleRay;
			Intersection sampleIntersection;
			
			sampleRay.origin = intersection.position;
			sampleRay.direction = sampleDirection;
			sampleRay.fastOcclusion = true;
			sampleRay.minDistance = scene.general.rayMinDistance;
			sampleRay.maxDistance = maxSampleDistance;
			sampleRay.precalculate();

			if (scene.intersect(sampleRay, sampleIntersection))
				occlusionCount++;
		}
	}

	float occlusionAmount = float(occlusionCount) / (float(n) * float(n));
	return (1.0f - occlusionAmount) * color;
}

Vector3 AmbientLight::getDirection(const Intersection& intersection) const
{
	(void)intersection;

	return Vector3();
}

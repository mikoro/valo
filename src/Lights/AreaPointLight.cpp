// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Lights/AreaPointLight.h"
#include "Tracing/Scene.h"
#include "Tracing/Intersection.h"
#include "Tracing/Ray.h"

using namespace Raycer;

void AreaPointLight::initialize()
{
	sampler = Sampler::getSampler(samplerType);
}

bool AreaPointLight::hasDirection() const
{
	return true;
}

Color AreaPointLight::getColor(const Scene& scene, const Intersection& intersection, Random& random) const
{
	Vector3 intersectionToLight = position - intersection.position;
	float distance2 = intersectionToLight.lengthSquared();
	float distance = std::sqrt(distance2);
	Vector3 directionToLight = intersectionToLight / distance;
	float cosine = intersection.normal.dot(directionToLight);

	Vector3 lightRight = directionToLight.cross(Vector3::ALMOST_UP).normalized();
	Vector3 lightUp = lightRight.cross(directionToLight).normalized();

	uint64_t occlusionCount = 0;
	uint64_t permutation = random.getUint64();
	uint64_t n = sampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector2 jitter = sampler->getDiscSample(x, y, n, n, permutation, random) * radius;
			Vector3 newLightPosition = position + jitter.x * lightRight + jitter.y * lightUp;
			Vector3 newIntersectionToLight = newLightPosition - intersection.position;
			float newDistance = newIntersectionToLight.length();
			Vector3 newDirectionToLight = newIntersectionToLight / newDistance;

			Ray sampleRay;
			Intersection sampleIntersection;

			sampleRay.origin = intersection.position;
			sampleRay.direction = newDirectionToLight;
			sampleRay.isShadowRay = true;
			sampleRay.fastOcclusion = true;
			sampleRay.minDistance = scene.general.rayMinDistance;
			sampleRay.maxDistance = newDistance - scene.general.rayMinDistance;
			sampleRay.precalculate();

			if (scene.intersect(sampleRay, sampleIntersection))
				occlusionCount++;
		}
	}

	float occlusionAmount = float(occlusionCount) / (float(n) * float(n));
	return (1.0f - occlusionAmount) * cosine * color / distance2;
}

Vector3 AreaPointLight::getDirection(const Intersection& intersection) const
{
	return (intersection.position - position).normalized();
}

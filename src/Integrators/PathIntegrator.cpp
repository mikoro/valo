// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Materials/Material.h"
#include "PathIntegrator.h"
#include "Utils/Random.h"

using namespace Raycer;

Color PathIntegrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random)
{
	Ray sampleRay = viewRay;
	Color color(0.0f, 0.0f, 0.0f);
	Color sampleBrdf(1.0f, 1.0f, 1.0f);
	float sampleCosine = 1.0f;
	float sampleProbability = 1.0f;
	float continuationProbability = 1.0f;
	uint32_t depth = 0;

	while (true)
	{
		Intersection intersection;
		scene.intersect(sampleRay, intersection);

		if (!intersection.wasFound)
			break;

		if (scene.general.normalMapping && intersection.material->normalTexture != nullptr)
			Integrator::calculateNormalMapping(intersection);

		if (depth == 0 && !intersection.isBehind && intersection.material->isEmissive())
		{
			color = intersection.material->getEmittance(intersection.texcoord, intersection.position);
			break;
		}

		if (depth++ >= minPathLength)
		{
			if (random.getFloat() < terminationProbability)
				break;

			continuationProbability *= (1.0f - terminationProbability);
		}

		Color directLight = Integrator::calculateDirectLight(scene, intersection, random);
		color += sampleBrdf * sampleCosine * directLight / sampleProbability / continuationProbability;

		Vector3 sampleDirection = intersection.material->getDirection(intersection, random);
		sampleBrdf *= intersection.material->getBrdf(intersection, sampleDirection);
		sampleCosine *= sampleDirection.dot(intersection.normal);
		sampleProbability *= intersection.material->getPdf(intersection, sampleDirection);

		if (sampleProbability == 0.0f)
			break;

		sampleRay = Ray();
		sampleRay.origin = intersection.position;
		sampleRay.direction = sampleDirection;
		sampleRay.minDistance = scene.general.rayMinDistance;
		sampleRay.precalculate();
	}

	return color;
}

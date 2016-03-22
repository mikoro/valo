// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Materials/Material.h"
#include "PathIntegrator.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color PathIntegrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random)
{
	Ray pathRay = viewRay;
	Color result(0.0f, 0.0f, 0.0f);
	
	for (uint32_t i = 0; i < scene.integrator.pathIntegrator.pathSamples; ++i)
	{
		Color throughput(1.0f, 1.0f, 1.0f);
		uint32_t pathLength = 0;

		for (;;)
		{
			Intersection intersection;
			
			if (!scene.intersect(pathRay, intersection))
			{
				result += throughput * scene.general.backgroundColor;
				break;
			}

			scene.calculateNormalMapping(intersection);

			if (++pathLength == 1 && !intersection.isBehind && intersection.material->isEmissive())
				result += throughput * intersection.material->getEmittance(intersection.texcoord, intersection.position);

			Vector3 in = -pathRay.direction;
			Vector3 out = intersection.material->getDirection(intersection, random);

			result += throughput * Integrator::calculateDirectLight(scene, intersection, in, random);

			Color brdf = intersection.material->getBrdf(intersection, in, out);
			float cosine = out.dot(intersection.normal);
			float pdf = intersection.material->getPdf(intersection, out);

			if (pdf == 0.0f)
				break;

			throughput *= brdf * cosine / pdf;

			if (pathLength >= minPathLength)
			{
				if (random.getFloat() < terminationProbability)
					break;

				throughput /= (1.0f - terminationProbability);
			}

			if (pathLength >= maxPathLength)
				break;

			pathRay = Ray();
			pathRay.origin = intersection.position;
			pathRay.direction = out;
			pathRay.minDistance = scene.general.rayMinDistance;
			pathRay.precalculate();
		}
	}

	return result / float(scene.integrator.pathIntegrator.pathSamples);
}

uint32_t PathIntegrator::getSampleCount() const
{
	return pathSamples;
}

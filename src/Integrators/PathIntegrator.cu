// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Materials/Material.h"
#include "Integrators/PathIntegrator.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color PathIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	Color result(0.0f, 0.0f, 0.0f);

	Color pathThroughput(1.0f, 1.0f, 1.0f);
	uint32_t pathLength = 0;
	Intersection pathIntersection = intersection;
	Ray pathRay = ray;

	for (;;)
	{
		const Material& material = scene.getMaterial(pathIntersection.materialIndex);

		if (++pathLength == 1 && !pathIntersection.isBehind && material.isEmissive())
			result += pathThroughput * material.getEmittance(scene, pathIntersection.texcoord, pathIntersection.position);

		Vector3 in = -pathRay.direction;
		Vector3 out = material.getDirection(pathIntersection, random);

		result += pathThroughput * Integrator::calculateDirectLight(scene, pathIntersection, in, random);

		Color brdf = material.getBrdf(scene, pathIntersection, in, out);
		float cosine = out.dot(pathIntersection.normal);
		float pdf = material.getPdf(pathIntersection, out);

		if (pdf == 0.0f)
			break;

		pathThroughput *= brdf * cosine / pdf;

		if (pathLength >= minPathLength)
		{
			if (random.getFloat() < terminationProbability)
				break;

			pathThroughput /= (1.0f - terminationProbability);
		}

		if (pathLength >= maxPathLength)
			break;

		pathRay = Ray();
		pathRay.origin = pathIntersection.position;
		pathRay.direction = out;
		pathRay.minDistance = scene.general.rayMinDistance;
		pathRay.precalculate();

		pathIntersection = Intersection();

		if (!scene.intersect(pathRay, pathIntersection))
			break;

		scene.calculateNormalMapping(pathIntersection);
	}

	return result;
}

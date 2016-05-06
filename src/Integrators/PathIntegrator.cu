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

	if (scene.getEmissiveTrianglesCount() == 0)
		return result;

	Color pathThroughput(1.0f, 1.0f, 1.0f);
	uint32_t pathLength = 0;
	Intersection pathIntersection = intersection;
	Ray pathRay = ray;

	for (;;)
	{
		++pathLength;
		const Material& material = scene.getMaterial(pathIntersection.materialIndex);

		if (pathLength == 1 && !pathIntersection.isBehind && material.showEmittance && material.isEmissive())
			result += pathThroughput * material.getEmittance(scene, pathIntersection.texcoord, pathIntersection.position);

		Vector3 in = -pathRay.direction;
		Vector3 out = material.getDirection(pathIntersection, random);

		Intersection emissiveIntersection = Integrator::getRandomEmissiveIntersection(scene, random);

		if (Integrator::isIntersectionVisible(scene, pathIntersection, emissiveIntersection))
		{
			DirectLightSample lightSample = Integrator::calculateDirectLightSample(scene, pathIntersection, emissiveIntersection);
			
			if (lightSample.visible && lightSample.lightPdf > 0.0f)
			{
				Color lightBrdf = material.getBrdf(scene, pathIntersection, in, lightSample.direction);
				float brdfPdf = material.getPdf(pathIntersection, lightSample.direction);
				float weight = Integrator::powerHeuristic(1, lightSample.lightPdf, 1, brdfPdf);
				result += pathThroughput * lightSample.emittance * lightBrdf * lightSample.originCosine * weight / lightSample.lightPdf;
			}
		}

		pathRay = Ray();
		pathRay.origin = pathIntersection.position;
		pathRay.direction = out;
		pathRay.minDistance = scene.general.rayMinDistance;
		pathRay.precalculate();

		Intersection previousPathIntersection = pathIntersection;
		pathIntersection = Intersection();

		if (!scene.intersect(pathRay, pathIntersection))
			break;

		scene.calculateNormalMapping(pathIntersection);

		const Material& nextMaterial = scene.getMaterial(pathIntersection.materialIndex);

		if (nextMaterial.isEmissive())
		{
			DirectLightSample lightSample = Integrator::calculateDirectLightSample(scene, previousPathIntersection, pathIntersection);

			if (lightSample.visible && lightSample.lightPdf > 0.0f)
			{
				Color lightBrdf = material.getBrdf(scene, previousPathIntersection, in, lightSample.direction);
				float brdfPdf = material.getPdf(previousPathIntersection, lightSample.direction);
				float weight = Integrator::powerHeuristic(1, lightSample.lightPdf, 1, brdfPdf);
				result += pathThroughput * lightSample.emittance * lightBrdf * lightSample.originCosine * weight / lightSample.lightPdf;
			}
		}

		Color pathBrdf = material.getBrdf(scene, previousPathIntersection, in, out);
		float pathCosine = out.dot(previousPathIntersection.normal);
		float pathPdf = material.getPdf(previousPathIntersection, out);

		if (pathCosine <= 0.0f || pathPdf <= 0.0f)
			break;

		pathThroughput *= pathBrdf * pathCosine / pathPdf;

		if (pathThroughput.isZero())
			break;

		if (pathLength >= maxPathLength)
			break;

		if (pathLength >= minPathLength)
		{
			if (random.getFloat() < terminationProbability)
				break;

			pathThroughput /= (1.0f - terminationProbability);
		}
	}

	return result;
}

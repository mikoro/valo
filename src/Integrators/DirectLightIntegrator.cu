// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/DirectLightIntegrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color DirectLightIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	const Material& material = scene.getMaterial(intersection.materialIndex);

	if (!intersection.isBehind && material.showEmittance && material.isEmissive())
		return material.getEmittance(scene, intersection.texcoord, intersection.position);

	Color result(0.0f, 0.0f, 0.0f);
	Intersection emissiveIntersection = Integrator::getRandomEmissiveIntersection(scene, random);

	if (Integrator::isIntersectionVisible(scene, intersection, emissiveIntersection))
	{
		DirectLightSample lightSample = Integrator::calculateDirectLightSample(scene, intersection, emissiveIntersection);

		if (lightSample.visible && lightSample.lightPdf > 0.0f)
		{
			Color lightBrdf = material.getBrdf(scene, intersection, -ray.direction, lightSample.direction);
			float brdfPdf = material.getPdf(intersection, lightSample.direction);
			float weight = Integrator::powerHeuristic(1, lightSample.lightPdf, 1, brdfPdf);
			result = lightSample.emittance * lightBrdf * lightSample.originCosine * weight / lightSample.lightPdf;
		}
	}

	return result;
}

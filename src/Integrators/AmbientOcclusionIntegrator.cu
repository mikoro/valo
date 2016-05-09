// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/AmbientOcclusionIntegrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"
#include "Math/Mapper.h"

using namespace Valo;

CUDA_CALLABLE Color AmbientOcclusionIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	(void)ray;

	Ray aoRay;

	aoRay.origin = intersection.position;
	aoRay.direction = Mapper::mapToCosineHemisphere(random.getVector2(), intersection.onb);
	aoRay.minDistance = scene.general.rayMinDistance;
	aoRay.maxDistance = maxDistance;
	aoRay.precalculate();

	Intersection aoIntersection;
	float aoValue = scene.intersect(aoRay, aoIntersection) ? 0.0f : 1.0f;
	aoValue *= std::abs(aoRay.direction.dot(intersection.normal));
	Color aoColor = Color(aoValue, aoValue, aoValue);

	if (useReflectance)
	{
		const Material& material = scene.getMaterial(intersection.materialIndex);
		aoColor *= material.getReflectance(scene, intersection.texcoord, intersection.position);
	}

	return aoColor;
}

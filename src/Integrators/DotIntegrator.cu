// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "DotIntegrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color DotIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	(void)random;

	const Material& material = scene.getMaterial(intersection.materialIndex);
	float dot = std::abs(ray.direction.dot(intersection.normal));
	return dot * material.getReflectance(scene, intersection.texcoord, intersection.position);
}

uint32_t DotIntegrator::getSampleCount() const
{
	return 1;
}

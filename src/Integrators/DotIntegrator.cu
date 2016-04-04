// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/DotIntegrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color DotIntegrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	(void)random;

	float dot = std::abs(ray.direction.dot(intersection.normal));
	Color dotColor = Color(dot, dot, dot);

	if (useReflectance)
	{
		const Material& material = scene.getMaterial(intersection.materialIndex);
		dotColor *= material.getReflectance(scene, intersection.texcoord, intersection.position);
	}

	return dotColor;
}

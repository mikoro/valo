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

CUDA_CALLABLE Color DotIntegrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random) const
{
	(void)random;

	Intersection intersection;

	if (!scene.intersect(viewRay, intersection))
		return scene.general.backgroundColor;

	if (intersection.hasColor)
		return intersection.color;

	scene.calculateNormalMapping(intersection);

	if (scene.general.normalVisualization)
		return Color::fromNormal(intersection.normal);

	const Material& material = scene.getMaterial(intersection.materialIndex);

	float dot = std::abs(viewRay.direction.dot(intersection.normal));
	return dot * material.getReflectance(scene, intersection.texcoord, intersection.position);
}

uint32_t DotIntegrator::getSampleCount() const
{
	return 1;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "DotIntegrator.h"
#include "Materials/Material.h"

using namespace Raycer;

Color DotIntegrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random)
{
	(void)random;

	Intersection intersection;
	scene.intersect(viewRay, intersection);

	if (!intersection.wasFound)
		return scene.general.backgroundColor;

	if (intersection.hasColor)
		return intersection.color;

	if (scene.general.normalMapping && intersection.material->normalTexture != nullptr)
		Integrator::calculateNormalMapping(intersection);

	if (scene.general.normalVisualization)
		return Color::fromNormal(intersection.normal);

	return intersection.material->getReflectance(intersection.texcoord, intersection.position) * std::abs(viewRay.direction.dot(intersection.normal));
}

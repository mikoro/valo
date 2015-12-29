// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PreviewTracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

using namespace Raycer;

Color PreviewTracer::trace(const Scene& scene, const Ray& ray, Random& random)
{
	(void)random;

	Color finalColor = scene.general.backgroundColor;
	Intersection intersection;

	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return finalColor;

	const Material* material = intersection.material;

	if (material->reflectanceMapTexture != nullptr)
		finalColor = material->reflectanceMapTexture->getColor(intersection.texcoord, intersection.position);
	else if (material->diffuseMapTexture != nullptr)
		finalColor = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position);
	else if (!material->reflectance.isZero())
		finalColor = material->reflectance;
	else if (!material->diffuseReflectance.isZero())
		finalColor = material->diffuseReflectance;

	return finalColor * std::abs(ray.direction.dot(intersection.normal));
}

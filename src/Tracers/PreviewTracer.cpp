// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stdafx.h"

#include "Tracers/PreviewTracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

using namespace Raycer;

Color PreviewTracer::trace(const Scene& scene, const Ray& ray, std::mt19937& generator, const std::atomic<bool>& interrupted)
{
	(void)generator;
	(void)interrupted;

	Color finalColor = Color::BLACK;
	Intersection intersection;

	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return finalColor;

	const Material* material = intersection.material;

	bool hasReflectance = !material->reflectance.isZero();

	if (hasReflectance)
	{
		finalColor = material->reflectance;

		if (material->reflectanceMapTexture != nullptr)
			finalColor = material->reflectanceMapTexture->getColor(intersection.texcoord, intersection.position) * material->reflectanceMapTexture->intensity;
	}
	else
	{
		finalColor = material->diffuseReflectance;

		if (material->diffuseMapTexture != nullptr)
			finalColor = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position) * material->diffuseMapTexture->intensity;
	}

	return finalColor * std::abs(ray.direction.dot(intersection.normal));
}

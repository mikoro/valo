// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PreviewTracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Rendering/Film.h"

using namespace Raycer;

uint64_t PreviewTracer::getPixelSampleCount(const Scene& scene) const
{
	(void)scene;

	return 1;
}

uint64_t PreviewTracer::getSamplesPerPixel(const Scene& scene) const
{
	(void)scene;

	return 1;
}

void PreviewTracer::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount)
{
	(void)random;

	++rayCount;
	pathCount = 0;

	bool isOffLens;
	Ray ray = scene.camera.getRay(pixelCenter, isOffLens);

	if (isOffLens)
	{
		film.addSample(pixelIndex, scene.general.offLensColor, 1.0f);
		return;
	}

	Intersection intersection;
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
	{
		film.addSample(pixelIndex, scene.general.backgroundColor, 1.0f);
		return;
	}

	Color color;
	Material* material = intersection.material;

	if (material->reflectanceMapTexture != nullptr)
		color = material->reflectanceMapTexture->getColor(intersection.texcoord, intersection.position);
	else if (material->diffuseMapTexture != nullptr)
		color = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position);
	else if (!material->reflectance.isZero())
		color = material->reflectance;
	else if (!material->diffuseReflectance.isZero())
		color = material->diffuseReflectance;

	color *= std::abs(ray.direction.dot(intersection.normal));
	film.addSample(pixelIndex, color, 1.0f);
}

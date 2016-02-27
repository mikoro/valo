// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/PreviewTracer.h"
#include "Tracing/Scene.h"
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

	if (intersection.hasColor)
	{
		film.addSample(pixelIndex, intersection.color, 1.0f);
		return;
	}

	if (scene.general.normalMapping && intersection.material->normalTexture != nullptr)
		calculateNormalMapping(intersection);

	if (scene.general.normalVisualization)
	{
		Color normalColor;

		normalColor.r = (intersection.normal.x + 1.0f) / 2.0f;
		normalColor.g = (intersection.normal.y + 1.0f) / 2.0f;
		normalColor.b = (intersection.normal.z + 1.0f) / 2.0f;

		film.addSample(pixelIndex, normalColor, 1.0f);
		return;
	}

	Color color = intersection.material->getReflectance(intersection);
	color *= std::abs(ray.direction.dot(intersection.normal));
	film.addSample(pixelIndex, color, 1.0f);
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <omp.h>

#include "Core/Film.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Core/Intersection.h"
#include "Renderers/CpuRenderer.h"
#include "Renderers/Renderer.h"

using namespace Raycer;

void CpuRenderer::initialize()
{
	omp_set_num_threads(maxThreadCount);
	uint32_t maxThreads = MAX(1, omp_get_max_threads());

	assert(maxThreads >= 1);

	if (maxThreads != randoms.size())
	{
		randoms.resize(maxThreads);
		std::random_device rd;
		std::mt19937_64 generator(rd());

		for (Random& random : randoms)
			random.seed(generator());
	}
}

void CpuRenderer::resize(uint32_t width, uint32_t height)
{
	(void)width;
	(void)height;
}

void CpuRenderer::render(RenderJob& job, bool filtering)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	std::mutex ompThreadExceptionMutex;
	std::exception_ptr ompThreadException = nullptr;

	const uint32_t filmWidth = film.getWidth();
	const uint32_t filmHeight = film.getHeight();
	const int32_t pixelCount = int32_t(filmWidth * filmHeight);

	#pragma omp parallel for schedule(dynamic, 1000)
	for (int32_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex)
	{
		try
		{
			for (uint32_t i = 0; i < scene.renderer.pixelSamples; ++i)
			{
				if ((pixelIndex + 1) % 100 == 0)
					job.totalSampleCount += 100 * scene.renderer.pixelSamples;

				if (job.interrupted)
					continue;

				float x = float(uint32_t(pixelIndex) % filmWidth);
				float y = float(uint32_t(pixelIndex) / filmWidth);

				Random& random = randoms[omp_get_thread_num()];

				Vector2 pixel = Vector2(x, y);
				float filterWeight = 1.0f;

				if (filtering && scene.renderer.filtering)
				{
					Vector2 offset = (random.getVector2() - Vector2(0.5f, 0.5f)) * 2.0f * scene.renderer.filter.getRadius();
					filterWeight = scene.renderer.filter.getWeight(offset);
					pixel += offset;
				}

				bool isOffLens;
				Ray ray = scene.camera.getRay(pixel, isOffLens);

				if (isOffLens)
				{
					film.addSample(pixelIndex, scene.general.offLensColor, filterWeight);
					continue;
				}

				Intersection intersection;

				if (!scene.intersect(ray, intersection))
				{
					film.addSample(pixelIndex, scene.general.backgroundColor, filterWeight);
					continue;
				}

				if (intersection.hasColor)
				{
					film.addSample(pixelIndex, intersection.color, filterWeight);
					continue;
				}

				scene.calculateNormalMapping(intersection);

				if (scene.general.normalVisualization)
				{
					film.addSample(pixelIndex, Color::fromNormal(intersection.normal), filterWeight);
					continue;
				}

				Color color = scene.integrator.calculateLight(scene, intersection, ray, random);
				film.addSample(pixelIndex, color, filterWeight);
			}
		}
		catch (...)
		{
			std::lock_guard<std::mutex> lock(ompThreadExceptionMutex);

			if (ompThreadException == nullptr)
				ompThreadException = std::current_exception();

			job.interrupted = true;
		}
	}

	if (ompThreadException != nullptr)
		std::rethrow_exception(ompThreadException);
}

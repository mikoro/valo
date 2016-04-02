// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <omp.h>

#include "Core/Film.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Renderers/CpuRenderer.h"
#include "Renderers/Renderer.h"

using namespace Raycer;

void CpuRenderer::initialize()
{
}

void CpuRenderer::render(RenderJob& job, bool filtering)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	omp_set_num_threads(maxThreadCount);
	uint32_t maxThreads = MAX(1, omp_get_max_threads());

	assert(maxThreads >= 1);

	if (maxThreads != randoms.size())
	{
		randoms.resize(maxThreads);
		std::random_device rd;

		for (Random& random : randoms)
			random.seed(uint64_t(rd()) << 32 | rd());
	}

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
			if (job.interrupted)
				continue;

			float x = float(uint32_t(pixelIndex) % filmWidth);
			float y = float(uint32_t(pixelIndex) / filmWidth);
			
			Vector2 pixel = Vector2(x, y);
			Random& random = randoms[omp_get_thread_num()];
			float filterWeight = 1.0f;
			
			if (filtering && scene.renderer.filtering)
			{
				Vector2 offset = (random.getVector2() - Vector2(0.5f, 0.5f)) * 2.0f * scene.renderer.filter.getRadius();
				filterWeight = scene.renderer.filter.getWeight(offset);
				pixel += offset;
			}

			bool isOffLens;
			Ray viewRay = scene.camera.getRay(pixel, isOffLens);

			if (isOffLens)
			{
				film.addSample(pixelIndex, scene.general.offLensColor, filterWeight);
				continue;
			}

			Color color = scene.integrator.calculateRadiance(scene, viewRay, random);
			film.addSample(pixelIndex, color, filterWeight);

			if ((pixelIndex + 1) % 100 == 0)
				job.sampleCount += 100 * scene.integrator.getSampleCount();
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

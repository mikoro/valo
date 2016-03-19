// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/CpuRenderer.h"
#include "Renderers/Renderer.h"

using namespace Raycer;

void CpuRenderer::initialize()
{
}

void CpuRenderer::render(RenderJob& job)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	omp_set_num_threads(maxThreadCount);
	uint64_t maxThreads = std::max(1, omp_get_max_threads());

	assert(maxThreads >= 1);

	if (maxThreads != randoms.size())
	{
		randoms.resize(maxThreads);

		for (Random& random : randoms)
			random.initialize();
	}

	std::mutex ompThreadExceptionMutex;
	std::exception_ptr ompThreadException = nullptr;

	const uint64_t filmWidth = film.getWidth();
	const uint64_t filmHeight = film.getHeight();
	const int64_t pixelCount = int64_t(filmWidth * filmHeight);

	#pragma omp parallel for schedule(dynamic, 1000)
	for (int64_t pixelIndex = 0; pixelIndex < pixelCount; ++pixelIndex)
	{
		try
		{
			if (job.interrupted)
				continue;

			float x = float(uint64_t(pixelIndex) % filmWidth);
			float y = float(uint64_t(pixelIndex) / filmWidth);
			
			Vector2 pixel = Vector2(x, y);
			Random& random = randoms[omp_get_thread_num()];
			float filterWeight = 1.0f;
			
			if (scene.general.pixelFiltering)
			{
				Vector2 offset = (random.getVector2() - Vector2(0.5f, 0.5f)) * 2.0f * scene.filter.getRadius();
				filterWeight = scene.filter.getWeight(offset);
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
				job.sampleCount += 100;
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

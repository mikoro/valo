// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Tracer.h"
#include "Tracers/Raytracer.h"
#include "Tracers/Pathtracer.h"
#include "Tracers/PreviewTracer.h"
#include "Scenes/Scene.h"
#include "App.h"
#include "Utils/Settings.h"
#include "TracerState.h"
#include "Rendering/Film.h"
#include "Samplers/CenterSampler.h"
#include "Samplers/RandomSampler.h"
#include "Samplers/RegularSampler.h"
#include "Samplers/JitteredSampler.h"
#include "Samplers/CMJSampler.h"
#include "Samplers/PoissonDiscSampler.h"
#include "Filters/BoxFilter.h"
#include "Filters/TentFilter.h"
#include "Filters/BellFilter.h"
#include "Filters/GaussianFilter.h"
#include "Filters/MitchellFilter.h"
#include "Filters/LanczosSincFilter.h"

using namespace Raycer;

std::unique_ptr<Tracer> Tracer::getTracer(TracerType type)
{
	switch (type)
	{
		case TracerType::RAY: return std::make_unique<Raytracer>();
		case TracerType::PATH: return std::make_unique<Pathtracer>();
		case TracerType::PREVIEW: return std::make_unique<PreviewTracer>();
		default: throw std::runtime_error("Invalid tracer type");
	}
}

Tracer::Tracer()
{
	samplers[SamplerType::CENTER] = std::make_unique<CenterSampler>();
	samplers[SamplerType::RANDOM] = std::make_unique<RandomSampler>();
	samplers[SamplerType::REGULAR] = std::make_unique<RegularSampler>();
	samplers[SamplerType::JITTERED] = std::make_unique<JitteredSampler>();
	samplers[SamplerType::CMJ] = std::make_unique<CMJSampler>();
	samplers[SamplerType::POISSON_DISC] = std::make_unique<PoissonDiscSampler>();

	filters[FilterType::BOX] = std::make_unique<BoxFilter>();
	filters[FilterType::TENT] = std::make_unique<TentFilter>();
	filters[FilterType::BELL] = std::make_unique<BellFilter>();
	filters[FilterType::GAUSSIAN] = std::make_unique<GaussianFilter>();
	filters[FilterType::MITCHELL] = std::make_unique<MitchellFilter>();
	filters[FilterType::LANCZOS_SINC] = std::make_unique<LanczosSincFilter>();
}

void Tracer::run(TracerState& state, std::atomic<bool>& interrupted)
{
	Settings& settings = App::getSettings();
	Scene& scene = *state.scene;
	Film& film = *state.film;

	omp_set_num_threads(settings.general.maxThreadCount);
	uint64_t maxThreads = std::max(1, omp_get_max_threads());

	assert(maxThreads >= 1);
	
	if (maxThreads != randoms.size())
	{
		randoms.resize(maxThreads);

		for (Random& random : randoms)
			random.initialize();
	}

	imageAutoWriteTimer.restart();
	filmAutoWriteTimer.restart();

	std::mutex ompThreadExceptionMutex;
	std::exception_ptr ompThreadException = nullptr;

	uint64_t pixelSampleCount = getPixelSampleCount(scene);
	uint64_t samplesPerPixel = getSamplesPerPixel(scene);

	for (uint64_t i = 0; i < pixelSampleCount && !interrupted; ++i)
	{
		#pragma omp parallel for schedule(dynamic, 1000)
		for (int64_t pixelIndex = 0; pixelIndex < int64_t(state.pixelCount); ++pixelIndex)
		{
			try
			{
				if (interrupted)
					continue;

				uint64_t offsetPixelIndex = uint64_t(pixelIndex) + state.pixelStartOffset;
				double x = double(offsetPixelIndex % state.filmWidth);
				double y = double(offsetPixelIndex / state.filmWidth);
				Vector2 pixelCenter = Vector2(x, y);
				Random& random = randoms[omp_get_thread_num()];

				trace(scene, film, pixelCenter, pixelIndex, random);

				if ((pixelIndex + 1) % 100 == 0)
					state.totalSamples += 100 * samplesPerPixel;
			}
			catch (...)
			{
				std::lock_guard<std::mutex> lock(ompThreadExceptionMutex);

				if (ompThreadException == nullptr)
					ompThreadException = std::current_exception();

				interrupted = true;
			}
		}

		if (ompThreadException != nullptr)
			std::rethrow_exception(ompThreadException);

		++state.pixelSamples;

		if (!settings.interactive.enabled)
		{
			if (settings.image.autoWrite && imageAutoWriteTimer.getElapsedSeconds() > settings.image.autoWriteInterval)
			{
				film.generateOutputImage(scene);
				film.getOutputImage().save(tfm::format(settings.image.autoWriteFileName.c_str(), imageAutoWriteNumber), false);

				if (++imageAutoWriteNumber > settings.image.autoWriteCount)
					imageAutoWriteNumber = 1;

				imageAutoWriteTimer.restart();
			}

			if (settings.film.autoWrite && filmAutoWriteTimer.getElapsedSeconds() > settings.film.autoWriteInterval)
			{
				film.save(tfm::format(settings.film.autoWriteFileName.c_str(), filmAutoWriteNumber), false);

				if (++filmAutoWriteNumber > settings.film.autoWriteCount)
					filmAutoWriteNumber = 1;

				filmAutoWriteTimer.restart();
			}
		}
	}

	if (!interrupted)
		state.totalSamples = state.pixelCount * pixelSampleCount * samplesPerPixel;
}

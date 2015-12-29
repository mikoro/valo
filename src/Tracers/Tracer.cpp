// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Tracer.h"
#include "Tracers/Raytracer.h"
#include "Tracers/Pathtracer.h"
#include "Tracers/PreviewTracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Rendering/Color.h"
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
	assert(scene.sampling.pixelSampleCount >= 1);

	if (maxThreads != randoms.size())
	{
		randoms.resize(maxThreads);

		for (Random& random : randoms)
			random.initialize();
	}

	std::mutex ompThreadExceptionMutex;
	std::exception_ptr ompThreadException = nullptr;

	imageAutoWriteTimer.restart();
	filmAutoWriteTimer.restart();

	for (uint64_t i = 0; i < scene.sampling.pixelSampleCount && !interrupted; ++i)
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
				Vector2 pixelCoordinate = Vector2(x, y);
				Random& random = randoms[omp_get_thread_num()];

				generateMultiSamples(scene, film, pixelCoordinate, uint64_t(pixelIndex), random);

				if ((pixelIndex + 1) % 100 == 0)
					state.pixelsProcessed += (100 / scene.sampling.pixelSampleCount);
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
		state.pixelsProcessed = state.pixelCount;
}

void Tracer::generateMultiSamples(const Scene& scene, Film& film, const Vector2& pixelCoordinate, uint64_t pixelIndex, Random& random)
{
	assert(scene.sampling.multiSampleCountSqrt >= 1);

	if (scene.sampling.multiSampleCountSqrt == 1)
	{
		Color pixelColor = generateTimeSamples(scene, pixelCoordinate, random);
		film.addSample(pixelIndex, pixelColor, 1.0);

		return;
	}
	
	Sampler* sampler = samplers[scene.sampling.multiSamplerType].get();
	Filter* filter = filters[scene.sampling.multiSamplerFilterType].get();

	uint64_t permutation = random.getUint64();
	uint64_t n = scene.sampling.multiSampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector2 sampleOffset = sampler->getSquareSample(x, y, n, n, permutation, random);
			sampleOffset = (sampleOffset - Vector2(0.5, 0.5)) * 2.0 * filter->getRadius();
			Color sampledPixelColor = generateTimeSamples(scene, pixelCoordinate + sampleOffset, random);
			film.addSample(pixelIndex, sampledPixelColor, filter->getWeight(sampleOffset));
		}
	}
}

Color Tracer::generateTimeSamples(const Scene& scene, const Vector2& pixelCoordinate, Random& random)
{
	assert(scene.sampling.timeSampleCount >= 1);

	if (scene.sampling.timeSampleCount == 1)
		return generateCameraSamples(scene, pixelCoordinate, 0.0, random);

	Sampler* sampler = samplers[scene.sampling.timeSamplerType].get();

	Color sampledPixelColor;
	uint64_t n = scene.sampling.timeSampleCount;

	for (uint64_t i = 0; i < n; ++i)
		sampledPixelColor += generateCameraSamples(scene, pixelCoordinate, sampler->getSample(i, n, 0, random), random);

	return sampledPixelColor / double(n);
}

Color Tracer::generateCameraSamples(const Scene& scene, const Vector2& pixelCoordinate, double time, Random& random)
{
	assert(scene.sampling.cameraSampleCountSqrt >= 1);

	Ray ray;
	ray.time = time;

	bool isValidRay = scene.camera.getRay(pixelCoordinate, ray);

	if (scene.sampling.cameraSampleCountSqrt == 1)
	{
		if (isValidRay)
			return trace(scene, ray, random);
		else
			return scene.general.offLensColor;
	}

	Sampler* sampler = samplers[scene.sampling.cameraSamplerType].get();

	uint64_t permutation = random.getUint64();

	double apertureSize = scene.camera.apertureSize;
	double focalDistance = scene.camera.focalDistance;

	Vector3 cameraPosition = scene.camera.position;
	Vector3 cameraRight = scene.camera.getRight();
	Vector3 cameraUp = scene.camera.getUp();

	Color sampledPixelColor;
	uint64_t n = scene.sampling.cameraSampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Ray primaryRay;
			primaryRay.time = time;

			Vector2 jitter = (sampler->getSquareSample(x, y, n, n, permutation, random) - Vector2(0.5, 0.5)) * 2.0;
			isValidRay = scene.camera.getRay(pixelCoordinate + jitter, primaryRay);

			if (!isValidRay)
			{
				sampledPixelColor += scene.general.offLensColor;
				continue;
			}

			Vector3 focalPoint = primaryRay.origin + primaryRay.direction * focalDistance;
			Vector2 discCoordinate = sampler->getDiscSample(x, y, n, n, permutation, random);

			Ray sampleRay;
			sampleRay.origin = cameraPosition + ((discCoordinate.x * apertureSize) * cameraRight + (discCoordinate.y * apertureSize) * cameraUp);
			sampleRay.direction = (focalPoint - sampleRay.origin).normalized();
			sampleRay.time = time;
			sampleRay.precalculate();

			sampledPixelColor += trace(scene, sampleRay, random);
		}
	}

	return sampledPixelColor / (double(n) * double(n));
}

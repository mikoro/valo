// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Tracer.h"
#include "Tracers/Raytracer.h"
#include "Tracers/PathtracerRecursive.h"
#include "Tracers/PathtracerIterative.h"
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
#include "Tracing/Intersection.h"
#include "Tracing/Ray.h"

using namespace Raycer;

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
		uint64_t rayCount = 0;
		uint64_t pathCount = 0;

		#pragma omp parallel for schedule(dynamic, 1000) reduction(+:rayCount) reduction(+:pathCount)
		for (int64_t pixelIndex = 0; pixelIndex < int64_t(state.filmPixelCount); ++pixelIndex)
		{
			try
			{
				if (interrupted)
					continue;

				uint64_t offsetPixelIndex = uint64_t(pixelIndex) + state.filmPixelOffset;
				float x = float(offsetPixelIndex % state.filmWidth);
				float y = float(offsetPixelIndex / state.filmWidth);
				Vector2 pixelCenter = Vector2(x, y);
				Random& random = randoms[omp_get_thread_num()];

				trace(scene, film, pixelCenter, pixelIndex, random, rayCount, pathCount);

				if ((pixelIndex + 1) % 100 == 0)
				{
					state.sampleCount += 100 * samplesPerPixel;
					state.pixelCount += 100;
					state.rayCount += rayCount;
					state.pathCount += pathCount;

					rayCount = 0;
					pathCount = 0;
				}
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

		++state.pixelSampleCount;
		state.rayCount += rayCount;
		state.pathCount += pathCount;

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
	{
		state.sampleCount = state.filmPixelCount * pixelSampleCount * samplesPerPixel;
		state.pixelCount = state.filmPixelCount;
	}
}

std::unique_ptr<Tracer> Tracer::getTracer(TracerType type)
{
	switch (type)
	{
		case TracerType::RAY: return std::make_unique<Raytracer>();
		case TracerType::PATH_RECURSIVE: return std::make_unique<PathtracerRecursive>();
		case TracerType::PATH_ITERATIVE: return std::make_unique<PathtracerIterative>();
		case TracerType::PREVIEW: return std::make_unique<PreviewTracer>();
		default: throw std::runtime_error("Invalid tracer type");
	}
}

void Tracer::calculateNormalMapping(Intersection& intersection)
{
	Color normalColor = intersection.material->normalMapTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

Color Tracer::calculateDirectLight(const Scene& scene, const Intersection& intersection, Random& random)
{
	uint64_t emitterCount = scene.emissiveTriangles.size();

	if (emitterCount == 0)
		return Color(0.0f, 0.0f, 0.0f);

	Triangle* emitter = scene.emissiveTriangles[random.getUint64(0, emitterCount - 1)];
	Intersection emitterIntersection = emitter->getRandomIntersection(random);
	Vector3 intersectionToEmitter = emitterIntersection.position - intersection.position;
	float emitterDistance2 = intersectionToEmitter.lengthSquared();
	float emitterDistance = sqrt(emitterDistance2);
	Vector3 sampleDirection = intersectionToEmitter / emitterDistance;

	Ray shadowRay;
	shadowRay.origin = intersection.position;
	shadowRay.direction = sampleDirection;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = emitterDistance - scene.general.rayMinDistance;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.precalculate();

	Intersection shadowIntersection;
	scene.intersect(shadowRay, shadowIntersection);

	if (shadowIntersection.wasFound)
		return Color(0.0f, 0.0f, 0.0f);

	float cosine1 = intersection.normal.dot(sampleDirection);
	float cosine2 = sampleDirection.dot(-emitter->normal);

	if (cosine1 < 0.0f || cosine2 < 0.0f)
		return Color(0.0f, 0.0f, 0.0f);

	float probability1 = 1.0f / float(emitterCount);
	float probability2 = 1.0f / emitter->getArea();
	
	Color emittance = emitter->material->getEmittance(emitterIntersection);
	Color intersectionBrdf = intersection.material->getBrdf(intersection, sampleDirection);

	return emittance * intersectionBrdf * cosine1 * cosine2 * (1.0f / emitterDistance2) / (probability1 * probability2);
}

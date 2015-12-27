// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "Samplers/Sampler.h"
#include "Filters/Filter.h"
#include "Rendering/Color.h"
#include "Utils/Random.h"
#include "Utils/Timer.h"

namespace Raycer
{
	struct TracerState;
	class Scene;
	class Film;
	class Ray;
	class Vector2;

	enum class TracerType { RAY, PATH, PREVIEW };

	class Tracer
	{
	public:

		Tracer();
		virtual ~Tracer() {}

		void run(TracerState& state, std::atomic<bool>& interrupted);

		static std::unique_ptr<Tracer> getTracer(TracerType type);

	protected:

		virtual Color trace(const Scene& scene, const Ray& ray, Random& random) = 0;

		std::map<SamplerType, std::unique_ptr<Sampler>> samplers;
		std::map<FilterType, std::unique_ptr<Filter>> filters;

	private:

		void generateMultiSamples(const Scene& scene, Film& film, const Vector2& pixelCoordinate, uint64_t pixelIndex, Random& random);
		Color generateTimeSamples(const Scene& scene, const Vector2& pixelCoordinate, Random& random);
		Color generateCameraSamples(const Scene& scene, const Vector2& pixelCoordinate, double time, Random& random);
		
		std::vector<Random> randoms;

		Timer imageAutoWriteTimer;
		Timer filmAutoWriteTimer;

		uint64_t imageAutoWriteNumber = 1;
		uint64_t filmAutoWriteNumber = 1;
	};
}

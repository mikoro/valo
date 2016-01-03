// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <map>
#include <memory>

#include "Samplers/Sampler.h"
#include "Filters/Filter.h"
#include "Utils/Random.h"
#include "Utils/Timer.h"

namespace Raycer
{
	struct TracerState;
	class Scene;
	class Film;
	class Ray;
	class Vector2;
	class Intersection;

	enum class TracerType { RAY, PATH_RECURSIVE, PATH_ITERATIVE, PREVIEW };

	class Tracer
	{
	public:

		Tracer();
		virtual ~Tracer() {}

		void run(TracerState& state, std::atomic<bool>& interrupted);

		virtual uint64_t getPixelSampleCount(const Scene& scene) const = 0;
		virtual uint64_t getSamplesPerPixel(const Scene& scene) const = 0;

		static std::unique_ptr<Tracer> getTracer(TracerType type);

	protected:

		virtual void trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& pathCount) = 0;

		static void calculateNormalMapping(Intersection& intersection);
		static Color calculateDirectLight(const Scene& scene, const Intersection& intersection, Random& random);

		std::map<SamplerType, std::unique_ptr<Sampler>> samplers;
		std::map<FilterType, std::unique_ptr<Filter>> filters;

	private:
		
		std::vector<Random> randoms;

		Timer imageAutoWriteTimer;
		Timer filmAutoWriteTimer;

		uint64_t imageAutoWriteNumber = 1;
		uint64_t filmAutoWriteNumber = 1;
	};
}

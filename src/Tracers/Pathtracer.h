// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Tracers/Tracer.h"
#include "Samplers/RandomSampler.h"

namespace Raycer
{
	struct TracerState;
	class Ray;
	
	class Pathtracer : public Tracer
	{
	protected:

		Color trace(const Scene& scene, const Ray& ray, Random& random) override;

	private:

		Color traceRecursive(const Scene& scene, const Ray& ray, Random& random);

		RandomSampler sampler;
	};
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <random>

#include "Tracers/Tracer.h"

namespace Raycer
{
	struct TracerState;
	class Ray;
	
	class PreviewTracer : public Tracer
	{
	protected:

		Color trace(const Scene& scene, const Ray& ray, std::mt19937& generator) override;
	};
}

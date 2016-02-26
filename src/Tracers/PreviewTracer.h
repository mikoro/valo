// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Tracers/Tracer.h"

namespace Raycer
{
	struct TracerState;
	class Ray;
	
	class PreviewTracer : public Tracer
	{
	public:

		uint64_t getPixelSampleCount(const Scene& scene) const override;
		uint64_t getSamplesPerPixel(const Scene& scene) const override;

	protected:

		void trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount) override;
	};
}

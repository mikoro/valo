// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/MovingAverage.h"
#include "Utils/Timer.h"
#include "Renderers/Renderer.h"

namespace Raycer
{
	class ConsoleRunner
	{
	public:

		int run();
		void interrupt();

	private:

		void printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint64_t pixelSamples);
		
		RenderJob renderJob;
		MovingAverage samplesPerSecondAverage;
	};
}

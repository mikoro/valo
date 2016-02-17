// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

#include "Math/MovingAverage.h"
#include "Utils/Timer.h"

namespace Raycer
{
	struct TracerState;

	class ConsoleRunner
	{
	public:

		int run();
		void run(TracerState& state);
		void interrupt();

	private:

		void printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint64_t pixelSamples);
		
		std::atomic<bool> interrupted;

		MovingAverage samplesPerSecondAverage;
		MovingAverage pixelsPerSecondAverage;
		MovingAverage raysPerSecondAverage;
		MovingAverage pathsPerSecondAverage;
	};
}

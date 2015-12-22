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

		void printProgress(const TimerData& elapsed, const TimerData& remaining);
		
		std::atomic<bool> interrupted;

		Timer timer;

		MovingAverage pixelsPerSecondAverage;
	};
}

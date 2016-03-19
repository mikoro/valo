// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/MovingAverage.h"
#include "Utils/Timer.h"
#include "Renderers/Renderer.h"

namespace Raycer
{
	class Scene;
	class Film;

	class ConsoleRunner
	{
	public:

		ConsoleRunner();
		~ConsoleRunner();

		int run();
		void interrupt();

	private:

		void printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint32_t pixelSamples);
		
		RenderJob renderJob;
		MovingAverage samplesPerSecondAverage;

		Scene* scene = nullptr;
		Film* film = nullptr;
	};
}

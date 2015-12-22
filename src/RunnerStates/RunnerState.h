// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	enum class RunnerStates { NONE, DEFAULT };

	class RunnerState
	{
	public:

		virtual ~RunnerState() {}

		virtual void initialize() = 0;
		virtual void pause() = 0;
		virtual void resume() = 0;
		virtual void shutdown() = 0;

		virtual void update(double timeStep) = 0;
		virtual void render(double timeStep, double interpolation) = 0;

		virtual void windowResized(uint64_t width, uint64_t height) = 0;
	};
}

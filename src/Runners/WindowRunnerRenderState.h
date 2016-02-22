// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <memory>

#include "Runners/WindowRunner.h"
#include "Tracers/Tracer.h"
#include "Tracing/Scene.h"
#include "Rendering/Film.h"
#include "Rendering/FilmRenderer.h"
#include "Rendering/InfoPanel.h"

namespace Raycer
{
	class WindowRunnerRenderState : public WindowRunnerState
	{
	public:

		WindowRunnerRenderState();

		void initialize() override;
		void shutdown() override;

		void update(float timeStep) override;
		void render(float timeStep, float interpolation) override;

		void windowResized(uint64_t width, uint64_t height) override;

	private:

		void resizeFilm();

		Scene scene;
		Film film;
		FilmRenderer filmRenderer;
		InfoPanel infoPanel;

		std::map<TracerType, std::unique_ptr<Tracer>> tracers;

		std::atomic<bool> interrupted;
		bool filmNeedsClearing = false;
	};
}

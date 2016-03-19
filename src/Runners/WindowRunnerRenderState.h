// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Runners/WindowRunner.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Utils/FilmQuad.h"
#include "Utils/InfoPanel.h"

namespace Raycer
{
	class WindowRunnerRenderState : public WindowRunnerState
	{
	public:

		void initialize() override;
		void shutdown() override;

		void update(float timeStep) override;
		void render(float timeStep, float interpolation) override;

		void windowResized(uint64_t width, uint64_t height) override;

	private:

		void resizeFilm();

		Renderer renderer;
		Scene scene;
		Film film;
		FilmQuad filmQuad;
		InfoPanel infoPanel;
	};
}

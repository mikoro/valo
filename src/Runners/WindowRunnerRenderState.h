// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Runners/WindowRunner.h"

#include "Core/Scene.h"
#include "Core/Film.h"
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

		void windowResized(uint32_t width, uint32_t height) override;

	private:

		void resizeFilm();

		Scene scene;
		Film film;
		Renderer renderer;
		FilmQuad filmQuad;
		InfoPanel infoPanel;
	};
}

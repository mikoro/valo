// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Scene.h"
#include "Core/Film.h"
#include "Renderers/Renderer.h"
#include "Utils/FilmQuad.h"
#include "Utils/InfoPanel.h"

namespace Valo
{
	class WindowRunnerRenderState
	{
	public:

		WindowRunnerRenderState();

		void initialize();
		void shutdown();

		void update(float timeStep);
		void render(float timeStep, float interpolation);

		void windowResized(uint32_t width, uint32_t height);

	private:

		void resizeFilm();

		Scene scene;
		Film film;
		Renderer renderer;
		FilmQuad filmQuad;
		InfoPanel infoPanel;
	};
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

struct NVGcontext;

namespace Raycer
{
	enum class InfoPanelState { OFF, FPS, FULL };

	class Scene;
	class Film;

	class InfoPanel
	{
	public:

		~InfoPanel();

		void initialize();
		void render(const Scene& scene, const Film& film);

		void setState(InfoPanelState state);
		void selectNextState();

	private:

		void renderFps();
		void renderFull(const Scene& scene, const Film& film);

		NVGcontext* context = nullptr;
		InfoPanelState currentState = InfoPanelState::OFF;
	};
}

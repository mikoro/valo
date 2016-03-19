// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

struct NVGcontext;

namespace Raycer
{
	struct RenderJob;

	enum class InfoPanelState { OFF, FPS, FULL };

	class InfoPanel
	{
	public:

		~InfoPanel();

		void initialize();
		void render(const Renderer& renderer, const RenderJob& job);

		void setState(InfoPanelState state);
		void selectNextState();

	private:

		void renderFps();
		void renderFull(const Renderer& renderer, const RenderJob& job);

		NVGcontext* context = nullptr;
		InfoPanelState currentState = InfoPanelState::OFF;
	};
}

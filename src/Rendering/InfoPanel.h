// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

struct NVGcontext;

namespace Raycer
{
	enum class InfoPanelState { OFF, FPS, FULL };

	struct TracerState;

	class InfoPanel
	{
	public:

		~InfoPanel();

		void initialize();
		void render(const TracerState& state);

		void setState(InfoPanelState state);
		void selectNextState();

	private:

		void renderFps();
		void renderFull(const TracerState& state);

		NVGcontext* context = nullptr;
		InfoPanelState currentState = InfoPanelState::OFF;
	};
}

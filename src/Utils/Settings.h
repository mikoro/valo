// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Settings
	{
	public:

		bool load(int argc, char** argv);

		struct Window
		{
			bool enabled;
			uint32_t width;
			uint32_t height;
			bool fullscreen;
			bool vsync;
			bool hideCursor;
			float renderScale;
			uint32_t infoPanelState;
			uint32_t infoPanelFontSize;
			bool checkGLErrors;
		} window;

		struct Image
		{
			uint32_t width;
			uint32_t height;
			std::string fileName;
			bool autoView;
		} image;

		struct Scene
		{
			std::string fileName;
			bool useTestScene;
			uint32_t testSceneNumber;
		} scene;

		struct Renderer
		{
			std::string fileName;
		} renderer;
	};
}

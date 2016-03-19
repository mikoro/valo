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
			uint64_t width;
			uint64_t height;
			bool fullscreen;
			bool vsync;
			bool hideCursor;
			float renderScale;
			uint64_t infoPanelState;
			uint64_t infoPanelFontSize;
			bool checkGLErrors;
		} window;

		struct Image
		{
			uint64_t width;
			uint64_t height;
			std::string fileName;
			bool autoView;
		} image;

		struct Scene
		{
			std::string fileName;
			bool useTestScene;
			uint64_t testSceneNumber;
		} scene;

		struct Renderer
		{
			std::string fileName;
		} renderer;
	};
}

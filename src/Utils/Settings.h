// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Settings
	{
	public:

		bool load(int argc, char** argv);

		struct General
		{
			bool windowed;
			uint32_t maxCpuThreadCount;
			uint32_t cudaDeviceNumber;
		} general;

		struct Renderer
		{
			uint32_t type;
			bool skip;
		} renderer;

		struct Window
		{
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

		struct Scene
		{
			std::string fileName;
			bool useTestScene;
			uint32_t testSceneNumber;
		} scene;

		struct Image
		{
			uint32_t width;
			uint32_t height;
			bool write;
			std::string fileName;
			bool autoView;
			bool autoWrite;
			float autoWriteInterval;
			std::string autoWriteFileName;
		} image;

		struct Film
		{
			bool load;
			std::string loadFileName;
			bool loadDir;
			std::string loadDirName;
			bool write;
			std::string writeFileName;
			bool autoView;
			bool autoWrite;
			float autoWriteInterval;
			std::string autoWriteFileName;
		} film;
	};
}

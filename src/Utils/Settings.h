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
			int32_t maxThreadCount;
		} general;

		struct Interactive
		{
			bool enabled;
			bool checkGLErrors;
			float renderScale;
			uint64_t infoPanelState;
			uint64_t infoPanelFontSize;
		} interactive;

		struct Scene
		{
			std::string fileName;
			bool enableTestScenes;
			uint64_t testSceneNumber;
		} scene;

		struct Image
		{
			uint64_t width;
			uint64_t height;
			std::string fileName;
			bool autoView;
			bool autoWrite;
			float autoWriteInterval;
			uint64_t autoWriteCount;
			std::string autoWriteFileName;
		} image;

		struct Window
		{
			uint64_t width;
			uint64_t height;
			bool enableFullscreen;
			bool enableVsync;
			bool hideCursor;
		} window;

		struct Film
		{
			bool restoreFromFile;
			std::string restoreFileName;
			bool autoWrite;
			float autoWriteInterval;
			uint64_t autoWriteCount;
			std::string autoWriteFileName;
		} film;

		struct Camera
		{
			bool enableMovement;
			bool smoothMovement;
			bool freeLook;
			float moveSpeed;
			float mouseSpeed;
			float moveDrag;
			float mouseDrag;
			float autoStopSpeed;
			float slowSpeedModifier;
			float fastSpeedModifier;
			float veryFastSpeedModifier;
		} camera;
	};
}

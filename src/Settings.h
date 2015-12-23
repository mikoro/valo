// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
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
			bool interactive;
			int32_t maxThreadCount;
			bool checkGLErrors;
		} general;

		struct Network
		{
			bool isClient;
			bool isServer;
			std::string localAddress;
			uint64_t localPort;
			std::string broadcastAddress;
			uint64_t broadcastPort;
		} network;

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
		} image;

		struct Window
		{
			uint64_t width;
			uint64_t height;
			double renderScale;
			bool enableFullscreen;
			bool enableVsync;
			bool hideCursor;
			bool showInfoText;
			std::string defaultFont;
			uint64_t defaultFontSize;
		} window;

		struct Camera
		{
			bool freeLook;
			bool smoothMovement;
			double moveSpeed;
			double mouseSpeed;
			double moveDrag;
			double mouseDrag;
			double autoStopSpeed;
			double slowSpeedModifier;
			double fastSpeedModifier;
			double veryFastSpeedModifier;
		} camera;
	};
}

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
			int32_t maxThreadCount;
		} general;

		struct Interactive
		{
			bool enabled;
			bool checkGLErrors;
			double renderScale;
			uint64_t infoPanelState;
			uint64_t infoPanelFontSize;
			bool usePreviewWhileMoving;
		} interactive;

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
			bool enableFullscreen;
			bool enableVsync;
			bool hideCursor;
		} window;

		struct Camera
		{
			bool enableMovement;
			bool smoothMovement;
			bool freeLook;
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

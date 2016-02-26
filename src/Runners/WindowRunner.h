// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <memory>

#include "Utils/FpsCounter.h"

struct GLFWwindow;

namespace Raycer
{
	enum class WindowRunnerStates { NONE, RENDER };

	class WindowRunnerState
	{
	public:

		virtual ~WindowRunnerState() {}

		virtual void initialize() = 0;
		virtual void shutdown() = 0;

		virtual void update(float timeStep) = 0;
		virtual void render(float timeStep, float interpolation) = 0;

		virtual void windowResized(uint64_t width, uint64_t height) = 0;
	};

	struct MouseInfo
	{
		int64_t windowX = 0;
		int64_t windowY = 0;
		int64_t filmX = 0;
		int64_t filmY = 0;
		int64_t deltaX = 0;
		int64_t deltaY = 0;
		float scrollY = 0.0f;
		bool hasScrolled = false;
	};

	class WindowRunner
	{
	public:

		WindowRunner();
		~WindowRunner();

		int run();
		void stop();

		GLFWwindow* getGlfwWindow() const;
		uint64_t getWindowWidth() const;
		uint64_t getWindowHeight() const;
		const MouseInfo& getMouseInfo() const;
		float getElapsedTime() const;
		const FpsCounter& getFpsCounter() const;

		bool keyIsDown(int32_t key);
		bool mouseIsDown(int32_t button);
		bool keyWasPressed(int32_t key);
		bool mouseWasPressed(int32_t button);
		float getMouseWheelScroll();

		void changeState(WindowRunnerStates state);

	private:

		void initialize();
		void shutdown();

		void checkWindowSize();
		void printWindowSize();
		void windowResized(uint64_t width, uint64_t height);
		
		void mainloop();
		void update(float timeStep);
		void render(float timeStep, float interpolation);

		void takeScreenshot() const;

		bool shouldRun = true;
		bool glfwInitialized = false;

		double startTime = 0.0;

		GLFWwindow* glfwWindow = nullptr;
		uint64_t windowWidth = 0;
		uint64_t windowHeight = 0;

		MouseInfo mouseInfo;
		int64_t previousMouseX = 0;
		int64_t previousMouseY = 0;

		std::map<int32_t, bool> keyStates;
		std::map<int32_t, bool> mouseStates;

		std::map<WindowRunnerStates, std::unique_ptr<WindowRunnerState>> windowRunnerStates;
		WindowRunnerState* currentState = nullptr;

		FpsCounter fpsCounter;
	};
}

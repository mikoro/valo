// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Runners/WindowRunner.h"
#include "App.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Utils/GLHelper.h"
#include "Rendering/Image.h"
#include "RunnerStates/RunnerState.h"
#include "RunnerStates/DefaultState.h"

using namespace Raycer;

namespace
{
	void glfwErrorCallback(int32_t error, const char* description)
	{
		App::getLog().logError("GLFW error (%s): %s", error, description);
	}

	MouseInfo* mouseInfoPtr = nullptr;

	void glfwMouseWheelScroll(GLFWwindow* window, double xoffset, double yoffset)
	{
		(void)window;
		(void)xoffset;

		mouseInfoPtr->scrollY = float(yoffset);
		mouseInfoPtr->hasScrolled = true;
	}
}

WindowRunner::WindowRunner()
{
}

WindowRunner::~WindowRunner()
{
	if (glfwInitialized)
		glfwTerminate();
}

int WindowRunner::run()
{
	initialize();
	mainLoop();
	shutdown();

	return 0;
}

void WindowRunner::stop()
{
	shouldRun = false;
}

void WindowRunner::pause()
{
	isPaused = true;
}

GLFWwindow* WindowRunner::getGlfwWindow() const
{
	return glfwWindow;
}

uint64_t WindowRunner::getWindowWidth() const
{
	return windowWidth;
}

uint64_t WindowRunner::getWindowHeight() const
{
	return windowHeight;
}

const MouseInfo& WindowRunner::getMouseInfo() const
{
	return mouseInfo;
}

float WindowRunner::getElapsedTime() const
{
	return float(glfwGetTime() - startTime);
}

const FpsCounter& WindowRunner::getFpsCounter() const
{
	return fpsCounter;
}

bool WindowRunner::keyIsDown(int32_t key)
{
	return (glfwGetKey(glfwWindow, key) == GLFW_PRESS);
}

bool WindowRunner::mouseIsDown(int32_t button)
{
	return (glfwGetMouseButton(glfwWindow, button) == GLFW_PRESS);
}

bool WindowRunner::keyWasPressed(int32_t key)
{
	if (keyIsDown(key))
	{
		if (!keyStates[key])
		{
			keyStates[key] = true;
			return true;
		}
	}
	else
		keyStates[key] = false;

	return false;
}

bool WindowRunner::mouseWasPressed(int32_t button)
{
	if (mouseIsDown(button))
	{
		if (!mouseStates[button])
		{
			mouseStates[button] = true;
			return true;
		}
	}
	else
		mouseStates[button] = false;

	return false;
}

float WindowRunner::getMouseWheelScroll()
{
	if (mouseInfo.hasScrolled)
	{
		mouseInfo.hasScrolled = false;
		return mouseInfo.scrollY;
	}

	return 0.0f;
}

void WindowRunner::changeState(RunnerStates newState)
{
	if (currentState != RunnerStates::NONE)
		runnerStates[currentState]->shutdown();

	currentState = newState;
	runnerStates[currentState]->initialize();
}

void WindowRunner::initialize()
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();

	mouseInfoPtr = &mouseInfo;

	log.logInfo("Initializing GLFW library");

	glfwSetErrorCallback(::glfwErrorCallback);

	if (!glfwInit())
		throw std::runtime_error("Could not initialize GLFW library");

	glfwInitialized = true;
	startTime = glfwGetTime();

	log.logInfo("Creating window and OpenGL context (%sx%s, fullscreen: %s)", settings.window.width, settings.window.height, settings.window.enableFullscreen);

#ifdef __APPLE__
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	glfwWindow = glfwCreateWindow(int32_t(settings.window.width), int32_t(settings.window.height), "Raycer", settings.window.enableFullscreen ? glfwGetPrimaryMonitor() : nullptr, nullptr);

	if (!glfwWindow)
		throw std::runtime_error("Could not create the window");

	printWindowSize();
	glfwSetScrollCallback(glfwWindow, ::glfwMouseWheelScroll);

	const GLFWvidmode* videoMode = glfwGetVideoMode(glfwGetPrimaryMonitor());
	glfwSetWindowPos(glfwWindow, (videoMode->width / 2 - int32_t(settings.window.width) / 2), (videoMode->height / 2 - int32_t(settings.window.height) / 2));
	glfwMakeContextCurrent(glfwWindow);

	log.logInfo("Initializing GL3W library");

	int32_t result = gl3wInit();

	if (result == -1)
		throw std::runtime_error("Could not initialize GL3W library");

	log.logInfo("OpenGL Vendor: %s | Renderer: %s | Version: %s | GLSL: %s", glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

	glfwSwapInterval(settings.window.enableVsync ? 1 : 0);

	if (settings.window.hideCursor)
		glfwSetInputMode(glfwWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	checkWindowSize();

	runnerStates[RunnerStates::DEFAULT] = std::make_unique<DefaultState>();
	changeState(RunnerStates::DEFAULT);
}

void WindowRunner::shutdown()
{
	if (currentState != RunnerStates::NONE)
		runnerStates[currentState]->shutdown();
}

void WindowRunner::checkWindowSize()
{
	int tempFramebufferWidth, tempFramebufferHeight;

	glfwGetFramebufferSize(glfwWindow, &tempFramebufferWidth, &tempFramebufferHeight);

	if (tempFramebufferWidth == 0 || tempFramebufferHeight == 0)
		return;

	if (uint64_t(tempFramebufferWidth) != windowWidth || uint64_t(tempFramebufferHeight) != windowHeight)
	{
		printWindowSize();
		windowResized(uint64_t(tempFramebufferWidth), uint64_t(tempFramebufferHeight));
	}
}

void WindowRunner::printWindowSize()
{
	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;

	glfwGetWindowSize(glfwWindow, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(glfwWindow, &tempFramebufferWidth, &tempFramebufferHeight);

	App::getLog().logInfo("GLFW window size: %dx%d | framebuffer size: %dx%d", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight);
}

void WindowRunner::windowResized(uint64_t width, uint64_t height)
{
	windowWidth = width;
	windowHeight = height;

	glViewport(0, 0, GLsizei(windowWidth), GLsizei(windowHeight));

	if (currentState != RunnerStates::NONE)
		runnerStates[currentState]->windowResized(windowWidth, windowHeight);
}

// http://gafferongames.com/game-physics/fix-your-timestep/
// http://gamesfromwithin.com/casey-and-the-clearly-deterministic-contraptions
// https://randomascii.wordpress.com/2012/02/13/dont-store-that-in-a-float/
void WindowRunner::mainLoop()
{
	App::getLog().logInfo("Entering the main loop");

	double timeStep = 1.0 / 60.0;
	double previousTime = glfwGetTime();
	double timeAccumulator = 0.0;

	update(0.0f);

	while (shouldRun)
	{
		double currentTime = glfwGetTime();
		double frameTime = currentTime - previousTime;
		previousTime = currentTime;

		// prevent too large frametimes (e.g. program was paused)
		if (frameTime > 0.25)
			frameTime = 0.25;

		timeAccumulator += frameTime;

		while (timeAccumulator >= timeStep)
		{
			update(float(timeStep));
			timeAccumulator -= timeStep;
		}

		double interpolation = timeAccumulator / timeStep;
		render(float(frameTime), float(interpolation));
	}
}

void WindowRunner::update(float timeStep)
{
	Settings& settings = App::getSettings();
	
	fpsCounter.update();
	glfwPollEvents();

	checkWindowSize();

	double newMouseX, newMouseY;
	glfwGetCursorPos(glfwWindow, &newMouseX, &newMouseY);

	mouseInfo.windowX = int64_t(newMouseX + 0.5);
	mouseInfo.windowY = int64_t(double(windowHeight) - newMouseY - 1.0 + 0.5);
	mouseInfo.filmX = int64_t((mouseInfo.windowX / double(windowWidth)) * (double(windowWidth) * settings.interactive.renderScale) + 0.5);
	mouseInfo.filmY = int64_t((mouseInfo.windowY / double(windowHeight)) * (double(windowHeight) * settings.interactive.renderScale) + 0.5);
	mouseInfo.deltaX = mouseInfo.windowX - previousMouseX;
	mouseInfo.deltaY = mouseInfo.windowY - previousMouseY;
	previousMouseX = mouseInfo.windowX;
	previousMouseY = mouseInfo.windowY;

	if (glfwWindowShouldClose(glfwWindow))
		shouldRun = false;

	if (keyWasPressed(GLFW_KEY_ESCAPE))
		shouldRun = false;

	if (keyWasPressed(GLFW_KEY_P))
		isPaused = !isPaused;

	if (currentState != RunnerStates::NONE)
	{
		if (!isPaused)
			runnerStates[currentState]->update(timeStep);
	}
	else
		throw std::runtime_error("Runner state has not been set");
}

void WindowRunner::render(float timeStep, float interpolation)
{
	fpsCounter.tick();

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	if (currentState != RunnerStates::NONE)
	{
		if (!isPaused)
			runnerStates[currentState]->render(timeStep, interpolation);
		else
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	else
		throw std::runtime_error("Runner state has not been set");

	glfwSwapBuffers(glfwWindow);

	if (keyWasPressed(GLFW_KEY_F12))
		takeScreenshot();
}

void WindowRunner::takeScreenshot() const
{
	std::vector<float> data(windowWidth * windowHeight * 4);

	glReadPixels(0, 0, GLsizei(windowWidth), GLsizei(windowHeight), GL_RGBA, GL_FLOAT, &data[0]);
	GLHelper::checkError("Could not read pixels from renderbuffer");

	Image image(windowWidth, windowHeight, &data[0]);
	image.save("screenshot.png");
}

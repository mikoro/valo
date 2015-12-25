﻿// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "RunnerStates/DefaultState.h"
#include "App.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Rendering/Color.h"
#include "Math/EulerAngle.h"
#include "Math/Vector3.h"
#include "Tracing/Camera.h"
#include "Tracers/TracerState.h"
#include "Rendering/ImagePool.h"
#include "Runners/WindowRunner.h"

using namespace Raycer;

DefaultState::DefaultState() : interrupted(false) {}

void DefaultState::initialize()
{
	Settings& settings = App::getSettings();

	if (settings.scene.enableTestScenes)
		scene = Scene::createTestScene(settings.scene.testSceneNumber);
	else
		scene = Scene::loadFromFile(settings.scene.fileName);

	filmRenderer.initialize();
	windowResized(settings.window.width, settings.window.height);

	scene.initialize();
	tracer = Tracer::getTracer(scene.general.tracerType);

	currentTestSceneNumber = settings.scene.testSceneNumber;
}

void DefaultState::pause()
{
}

void DefaultState::resume()
{
}

void DefaultState::shutdown()
{
}

void DefaultState::update(double timeStep)
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	bool increaseTestSceneNumber = windowRunner.keyWasPressed(GLFW_KEY_F2);
	bool decreaseTestSceneNumber = windowRunner.keyWasPressed(GLFW_KEY_F3);

	if (increaseTestSceneNumber || decreaseTestSceneNumber)
	{
		uint64_t previousTestSceneNumber = currentTestSceneNumber;

		if (increaseTestSceneNumber)
			currentTestSceneNumber++;

		if (decreaseTestSceneNumber)
			currentTestSceneNumber--;

		if (currentTestSceneNumber < 1)
			currentTestSceneNumber = 1;

		if (currentTestSceneNumber > Scene::TEST_SCENE_COUNT)
			currentTestSceneNumber = Scene::TEST_SCENE_COUNT;

		if (previousTestSceneNumber != currentTestSceneNumber)
		{
			ImagePool::clear();

			try
			{
				scene = Scene::createTestScene(currentTestSceneNumber);
				scene.initialize();
			}
			catch (const std::exception& ex)
			{
				log.logWarning("Could not create test scene: %s", ex.what());

				scene = Scene();
				scene.initialize();
			}

			scene.camera.setImagePlaneSize(film.getWidth(), film.getHeight());
			tracer = Tracer::getTracer(scene.general.tracerType);
			sampleCount = 0;
		}
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_R))
	{
		scene.camera.reset();
		film.clear();
		sampleCount = 0;
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
		scene.general.enableNormalMapping = !scene.general.enableNormalMapping;

	if (windowRunner.keyWasPressed(GLFW_KEY_F4))
	{
		if (scene.general.tracerType == TracerType::RAY)
			scene.general.tracerType = TracerType::PATH;
		else if (scene.general.tracerType == TracerType::PATH)
			scene.general.tracerType = TracerType::PREVIEW;
		else if (scene.general.tracerType == TracerType::PREVIEW)
			scene.general.tracerType = TracerType::RAY;

		tracer = Tracer::getTracer(scene.general.tracerType);
		film.clear();
		sampleCount = 0;
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F5))
	{
		if (scene.tonemapper.type == TonemapperType::PASSTHROUGH)
		{
			scene.tonemapper.type = TonemapperType::LINEAR;
			log.logInfo("Selected linear tone mapper");
		}
		else if (scene.tonemapper.type == TonemapperType::LINEAR)
		{
			scene.tonemapper.type = TonemapperType::SIMPLE;
			log.logInfo("Selected simple tone mapper");
		}
		else if (scene.tonemapper.type == TonemapperType::SIMPLE)
		{
			scene.tonemapper.type = TonemapperType::REINHARD;
			log.logInfo("Selected reinhard tone mapper");
		}
		else if (scene.tonemapper.type == TonemapperType::REINHARD)
		{
			scene.tonemapper.type = TonemapperType::PASSTHROUGH;
			log.logInfo("Selected passthrough tone mapper");
		}
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F7))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL))
		{
			if (windowRunner.keyIsDown(GLFW_KEY_LEFT_SHIFT))
				scene.saveToFile("scene.bin");
			else
				scene.saveToFile("scene.json");
		}
		else
			scene.saveToFile("scene.xml");
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F8))
	{
		windowRunner.pause();
		scene.saveToFile("scene.bin");

#ifdef _WIN32
		ShellExecuteA(nullptr, "open", "raycer.exe", "--scene.fileName scene.bin --general.interactive 0 --scene.enableTestScenes 0 --image.autoView 1", nullptr, SW_SHOWNORMAL);
#else
		int32_t pid = fork();

		if (pid == 0)
		{
			char* arg[] = { (char*)"raycer", (char*)"--scene.fileName scene.bin", (char*)"--general.interactive 0", (char*)"--scene.enableTestScenes 0", (char*)"--image.autoView 1", (char*)nullptr };

			if (execvp(arg[0], arg) == -1)
				App::getLog().logWarning("Could not launch external rendering (%d) (try adding raycer to PATH)", errno);
		}
#endif
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F10))
	{
		double newScale = settings.window.renderScale * 0.5;
		uint64_t newWidth = uint64_t(double(windowRunner.getWindowWidth()) * newScale + 0.5);
		uint64_t newHeight = uint64_t(double(windowRunner.getWindowHeight()) * newScale + 0.5);

		if (newWidth >= 2 && newHeight >= 2)
		{
			settings.window.renderScale = newScale;
			resizeFilm();
		}
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F11))
	{
		if (settings.window.renderScale < 1.0)
		{
			settings.window.renderScale *= 2.0;

			if (settings.window.renderScale > 1.0)
				settings.window.renderScale = 1.0;

			resizeFilm();
		}
	}

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			scene.tonemapper.exposure -= 2.0 * timeStep;
			scene.tonemapper.key -= 0.5 * timeStep;
		}
		else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			scene.tonemapper.exposure += 2.0 * timeStep;
			scene.tonemapper.key += 0.5 * timeStep;
		}

		scene.tonemapper.key = std::max(0.0, scene.tonemapper.key);
	}

	scene.camera.update(timeStep);
}

void DefaultState::render(double timeStep, double interpolation)
{
	(void)timeStep;
	(void)interpolation;

	Settings& settings = App::getSettings();
	
	TracerState state;
	state.scene = &scene;
	state.film = &film;
	state.filmWidth = film.getWidth();
	state.filmHeight = film.getHeight();
	state.pixelStartOffset = 0;
	state.pixelCount = state.filmWidth * state.filmHeight;

	if (scene.general.tracerType == TracerType::RAY
		|| scene.general.tracerType == TracerType::PREVIEW
		|| (scene.general.tracerType == TracerType::PATH && scene.camera.isMoving()))
	{
		film.clear();
		sampleCount = 0;
	}

	sampleCount += scene.general.pathSampleCount;

	tracer->run(state, interrupted);
	film.generateOutput(scene);
	filmRenderer.uploadFilmData(film);
	filmRenderer.render();

	if (settings.window.showInfoText)
	{
		/*int64_t scaledMouseX = windowRunner.getMouseInfo().scaledX;
		int64_t scaledMouseY = windowRunner.getMouseInfo().scaledY;
		int64_t scaledMouseIndex = scaledMouseY * film.getWidth() + scaledMouseX;*/
	}
}

void DefaultState::windowResized(uint64_t width, uint64_t height)
{
	filmRenderer.setWindowSize(width, height);
	resizeFilm();
}

void DefaultState::resizeFilm()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	uint64_t filmWidth = uint64_t(double(windowRunner.getWindowWidth()) * settings.window.renderScale + 0.5);
	uint64_t filmHeight = uint64_t(double(windowRunner.getWindowHeight()) * settings.window.renderScale + 0.5);

    filmWidth = std::max(uint64_t(1), filmWidth);
    filmHeight = std::max(uint64_t(1), filmHeight);

	film.resize(filmWidth, filmHeight);
	filmRenderer.setFilmSize(filmWidth, filmHeight);
	scene.camera.setImagePlaneSize(filmWidth, filmHeight);
	sampleCount = 0;
}

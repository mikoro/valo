// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "RunnerStates/DefaultState.h"
#include "App.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Tracing/Camera.h"
#include "Tracers/TracerState.h"
#include "Tracers/Raytracer.h"
#include "Tracers/PathtracerRecursive.h"
#include "Tracers/PathtracerIterative.h"
#include "Tracers/PreviewTracer.h"
#include "Runners/WindowRunner.h"

using namespace Raycer;

DefaultState::DefaultState() : interrupted(false)
{
	tracers[TracerType::RAY] = std::make_unique<Raytracer>();
	tracers[TracerType::PATH_RECURSIVE] = std::make_unique<PathtracerRecursive>();
	tracers[TracerType::PATH_ITERATIVE] = std::make_unique<PathtracerIterative>();
	tracers[TracerType::PREVIEW] = std::make_unique<PreviewTracer>();
}

void DefaultState::initialize()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	if (settings.scene.enableTestScenes)
		scene = Scene::createTestScene(settings.scene.testSceneNumber);
	else
		scene = Scene::loadFromFile(settings.scene.fileName);

	currentTestSceneNumber = settings.scene.testSceneNumber;

	filmRenderer.initialize();
	windowResized(windowRunner.getWindowWidth(), windowRunner.getWindowHeight());

	if (settings.film.restoreFromFile)
		film.load(settings.film.restoreFileName);

	scene.initialize();

	infoPanel.initialize();
	infoPanel.setState(InfoPanelState(settings.interactive.infoPanelState));
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

	if (windowRunner.keyWasPressed(GLFW_KEY_F1))
		infoPanel.selectNextState();

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
			film.clear();
		}
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_R))
	{
		scene.camera.reset();
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
		scene.general.enableNormalMapping = !scene.general.enableNormalMapping;

	if (windowRunner.keyWasPressed(GLFW_KEY_F4))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
			settings.interactive.usePreviewWhileMoving = !settings.interactive.usePreviewWhileMoving;
		else
		{
			if (scene.general.tracerType == TracerType::RAY)
				scene.general.tracerType = TracerType::PATH_RECURSIVE;
			else if (scene.general.tracerType == TracerType::PATH_RECURSIVE)
				scene.general.tracerType = TracerType::PATH_ITERATIVE;
			else if (scene.general.tracerType == TracerType::PATH_ITERATIVE)
				scene.general.tracerType = TracerType::PREVIEW;
			else if (scene.general.tracerType == TracerType::PREVIEW)
				scene.general.tracerType = TracerType::RAY;
		}
		
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F5))
	{
		if (scene.tonemapping.type == TonemapperType::PASSTHROUGH)
			scene.tonemapping.type = TonemapperType::LINEAR;
		else if (scene.tonemapping.type == TonemapperType::LINEAR)
			scene.tonemapping.type = TonemapperType::SIMPLE;
		else if (scene.tonemapping.type == TonemapperType::SIMPLE)
			scene.tonemapping.type = TonemapperType::REINHARD;
		else if (scene.tonemapping.type == TonemapperType::REINHARD)
			scene.tonemapping.type = TonemapperType::PASSTHROUGH;
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F6))
	{
		windowRunner.pause();
		scene.saveToFile("temp_scene.bin");

#ifdef _WIN32
		ShellExecuteA(nullptr, "open", "raycer.exe", "--scene.fileName temp_scene.bin --general.interactive 0 --scene.enableTestScenes 0 --image.autoView 1", nullptr, SW_SHOWNORMAL);
#else
		int32_t pid = fork();

		if (pid == 0)
		{
			char* arg[] = { (char*)"raycer", (char*)"--scene.fileName temp_scene.bin", (char*)"--general.interactive 0", (char*)"--scene.enableTestScenes 0", (char*)"--image.autoView 1", (char*)nullptr };

			if (execvp(arg[0], arg) == -1)
				App::getLog().logWarning("Could not launch external rendering (%d) (try adding raycer to PATH)", errno);
		}
#endif
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F10))
	{
		double newScale = settings.interactive.renderScale * 0.5;
		uint64_t newWidth = uint64_t(double(windowRunner.getWindowWidth()) * newScale + 0.5);
		uint64_t newHeight = uint64_t(double(windowRunner.getWindowHeight()) * newScale + 0.5);

		if (newWidth >= 2 && newHeight >= 2)
		{
			settings.interactive.renderScale = newScale;
			resizeFilm();
		}
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F11))
	{
		if (settings.interactive.renderScale < 1.0)
		{
			settings.interactive.renderScale *= 2.0;

			if (settings.interactive.renderScale > 1.0)
				settings.interactive.renderScale = 1.0;

			resizeFilm();
		}
	}

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			scene.tonemapping.exposure -= 2.0 * timeStep;
			scene.tonemapping.key -= 0.5 * timeStep;
		}
		else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			scene.tonemapping.exposure += 2.0 * timeStep;
			scene.tonemapping.key += 0.5 * timeStep;
		}

		scene.tonemapping.key = std::max(0.0, scene.tonemapping.key);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_M))
		settings.camera.enableMovement = !settings.camera.enableMovement;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_1))
			scene.saveToFile("temp_scene.xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_2))
			scene.saveToFile("temp_scene.json");

		if (windowRunner.keyWasPressed(GLFW_KEY_3))
			scene.saveToFile("temp_scene.bin");

		if (windowRunner.keyWasPressed(GLFW_KEY_4))
			scene.camera.saveState("camera.txt");

		if (windowRunner.keyWasPressed(GLFW_KEY_5))
		{
			film.generateOutputImage(scene);
			film.getOutputImage().save("temp_result.png");
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_6))
		{
			film.generateOutputImage(scene);
			film.getOutputImage().save("temp_result.hdr");
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_7))
			film.save("temp_film.bin");
	}

	scene.camera.update(timeStep);
}

void DefaultState::render(double timeStep, double interpolation)
{
	(void)timeStep;
	(void)interpolation;

	Settings& settings = App::getSettings();

	if (scene.general.tracerType == TracerType::RAY ||
		scene.general.tracerType == TracerType::PREVIEW ||
		((scene.general.tracerType == TracerType::PATH_RECURSIVE || scene.general.tracerType == TracerType::PATH_ITERATIVE) && scene.camera.isMoving()) ||
		filmNeedsClearing)
	{
		film.clear();
		filmNeedsClearing = false;
	}

	Tracer* tracer = tracers[scene.general.tracerType].get();

	if ((scene.general.tracerType == TracerType::PATH_RECURSIVE || scene.general.tracerType == TracerType::PATH_ITERATIVE) &&
		settings.interactive.usePreviewWhileMoving &&
		scene.camera.isMoving())
	{
		tracer = tracers[TracerType::PREVIEW].get();
		filmNeedsClearing = true;
	}

	uint64_t samplesPerPixel = tracer->getPixelSampleCount(scene) * tracer->getSamplesPerPixel(scene);
	film.increasePixelSamples(samplesPerPixel);

	TracerState state;
	state.scene = &scene;
	state.film = &film;
	state.filmWidth = film.getWidth();
	state.filmHeight = film.getHeight();
	state.pixelStartOffset = 0;
	state.pixelCount = state.filmWidth * state.filmHeight;
	
	tracer->run(state, interrupted);

	film.generateOutputImage(scene);
	filmRenderer.uploadFilmData(film);
	filmRenderer.render();
	infoPanel.render(state);
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

	uint64_t filmWidth = uint64_t(double(windowRunner.getWindowWidth()) * settings.interactive.renderScale + 0.5);
	uint64_t filmHeight = uint64_t(double(windowRunner.getWindowHeight()) * settings.interactive.renderScale + 0.5);

    filmWidth = std::max(uint64_t(1), filmWidth);
    filmHeight = std::max(uint64_t(1), filmHeight);

	film.resize(filmWidth, filmHeight);
	filmRenderer.setFilmSize(filmWidth, filmHeight);
	scene.camera.setImagePlaneSize(filmWidth, filmHeight);
}

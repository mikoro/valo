// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Runners/WindowRunnerRenderState.h"
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
#include "TestScenes/TestScene.h"

using namespace Raycer;

WindowRunnerRenderState::WindowRunnerRenderState() : interrupted(false)
{
	tracers[TracerType::RAY] = std::make_unique<Raytracer>();
	tracers[TracerType::PATH_RECURSIVE] = std::make_unique<PathtracerRecursive>();
	tracers[TracerType::PATH_ITERATIVE] = std::make_unique<PathtracerIterative>();
	tracers[TracerType::PREVIEW] = std::make_unique<PreviewTracer>();
}

void WindowRunnerRenderState::initialize()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	if (settings.scene.enableTestScenes)
		scene = TestScene::create(settings.scene.testSceneNumber);
	else
		scene = Scene::loadFromFile(settings.scene.fileName);

	filmRenderer.initialize();
	windowResized(windowRunner.getWindowWidth(), windowRunner.getWindowHeight());

	if (settings.film.restoreFromFile)
		film.load(settings.film.restoreFileName);

	scene.initialize();

	infoPanel.initialize();
	infoPanel.setState(InfoPanelState(settings.interactive.infoPanelState));
}

void WindowRunnerRenderState::shutdown()
{
}

void WindowRunnerRenderState::update(float timeStep)
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	bool ctrlIsPressed = windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL);
	
	// INFO PANEL //

	if (!ctrlIsPressed && windowRunner.keyWasPressed(GLFW_KEY_F1))
		infoPanel.selectNextState();

	// TRACER //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
			scene.general.tracerType = TracerType::RAY;

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			scene.general.tracerType = TracerType::PATH_RECURSIVE;
			film.clear();
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
			scene.general.tracerType = TracerType::PREVIEW;
	}

	// TONEMAPPER //

	if (!ctrlIsPressed && windowRunner.keyWasPressed(GLFW_KEY_F5))
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

	// RENDER SCALE //

	if (!ctrlIsPressed && windowRunner.keyWasPressed(GLFW_KEY_F6))
	{
		float newScale = settings.interactive.renderScale * 0.5f;
		uint64_t newWidth = uint64_t(float(windowRunner.getWindowWidth()) * newScale + 0.5f);
		uint64_t newHeight = uint64_t(float(windowRunner.getWindowHeight()) * newScale + 0.5f);

		if (newWidth >= 2 && newHeight >= 2)
		{
			settings.interactive.renderScale = newScale;
			resizeFilm();
		}
	}

	if (!ctrlIsPressed && windowRunner.keyWasPressed(GLFW_KEY_F7))
	{
		if (settings.interactive.renderScale < 1.0f)
		{
			settings.interactive.renderScale *= 2.0f;

			if (settings.interactive.renderScale > 1.0f)
				settings.interactive.renderScale = 1.0f;

			resizeFilm();
		}
	}

	// MISC //
	
	if (windowRunner.keyWasPressed(GLFW_KEY_R))
	{
		scene.camera.reset();
		film.clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
		scene.general.enableNormalMapping = !scene.general.enableNormalMapping;

	if (windowRunner.keyWasPressed(GLFW_KEY_M))
		settings.camera.enableMovement = !settings.camera.enableMovement;

	// EXPOSURE & KEY //

	if (ctrlIsPressed)
	{
		if (scene.tonemapping.type == TonemapperType::REINHARD)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene.tonemapping.key -= 0.1f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene.tonemapping.key += 0.1f * timeStep;

			scene.tonemapping.key = std::max(0.0f, scene.tonemapping.key);
		}
		else
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene.tonemapping.exposure -= 2.0f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene.tonemapping.exposure += 2.0f * timeStep;
		}
	}

	// SCENE/CAMERA/FILM SAVING //

	if (ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F1))
			scene.saveToFile("scene.xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
			scene.saveToFile("scene.json");

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
			scene.camera.saveState("camera.txt");

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			film.generateOutputImage(scene);
			film.getOutputImage().save("output.png");
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F5))
			film.save("film.bin");

		if (windowRunner.keyWasPressed(GLFW_KEY_F6))
			scene.saveBvhData("bvh.bin");

		if (windowRunner.keyWasPressed(GLFW_KEY_F7))
			scene.saveImagePool("imagepool.bin");
	}

	// TEST SCENE LOADING //

	int64_t testSceneIndex = -1;

	if (windowRunner.keyWasPressed(GLFW_KEY_1)) testSceneIndex = 1;
	if (windowRunner.keyWasPressed(GLFW_KEY_2)) testSceneIndex = 2;
	if (windowRunner.keyWasPressed(GLFW_KEY_3)) testSceneIndex = 3;
	if (windowRunner.keyWasPressed(GLFW_KEY_4)) testSceneIndex = 4;
	if (windowRunner.keyWasPressed(GLFW_KEY_5)) testSceneIndex = 5;
	if (windowRunner.keyWasPressed(GLFW_KEY_6)) testSceneIndex = 6;
	if (windowRunner.keyWasPressed(GLFW_KEY_7)) testSceneIndex = 7;
	if (windowRunner.keyWasPressed(GLFW_KEY_8)) testSceneIndex = 8;
	if (windowRunner.keyWasPressed(GLFW_KEY_9)) testSceneIndex = 9;
	if (windowRunner.keyWasPressed(GLFW_KEY_0)) testSceneIndex = 10;

	if (testSceneIndex != -1)
	{
		if (ctrlIsPressed)
			testSceneIndex += 10;

		try
		{
			scene = TestScene::create(testSceneIndex);
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

	scene.camera.update(timeStep);
}

void WindowRunnerRenderState::render(float timeStep, float interpolation)
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
	film.increasePixelSampleCount(samplesPerPixel);

	TracerState state;
	state.scene = &scene;
	state.film = &film;
	state.filmWidth = film.getWidth();
	state.filmHeight = film.getHeight();
	state.filmPixelOffset = 0;
	state.filmPixelCount = state.filmWidth * state.filmHeight;
	
	tracer->run(state, interrupted);

	film.generateOutputImage(scene);
	filmRenderer.uploadFilmData(film);
	filmRenderer.render();
	infoPanel.render(state);
}

void WindowRunnerRenderState::windowResized(uint64_t width, uint64_t height)
{
	filmRenderer.setWindowSize(width, height);
	resizeFilm();
}

void WindowRunnerRenderState::resizeFilm()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	uint64_t filmWidth = uint64_t(float(windowRunner.getWindowWidth()) * settings.interactive.renderScale + 0.5);
	uint64_t filmHeight = uint64_t(float(windowRunner.getWindowHeight()) * settings.interactive.renderScale + 0.5);

    filmWidth = std::max(uint64_t(1), filmWidth);
    filmHeight = std::max(uint64_t(1), filmHeight);

	film.resize(filmWidth, filmHeight);
	filmRenderer.setFilmSize(filmWidth, filmHeight);
	scene.camera.setImagePlaneSize(filmWidth, filmHeight);
}

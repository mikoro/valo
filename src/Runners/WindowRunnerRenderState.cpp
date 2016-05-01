// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <GLFW/glfw3.h>

#include "App.h"
#include "Core/Camera.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Runners/WindowRunner.h"
#include "Runners/WindowRunnerRenderState.h"
#include "TestScenes/TestScene.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"

using namespace Raycer;

WindowRunnerRenderState::WindowRunnerRenderState() : film(true)
{
}

void WindowRunnerRenderState::initialize()
{
	Settings& settings = App::getSettings();
	
	scene = TestScene::create(settings.scene.testSceneNumber);
	scene.initialize();
	film.initialize();
	renderer.initialize(settings);
	filmQuad.initialize();
	infoPanel.initialize();
	infoPanel.setState(InfoPanelState(settings.window.infoPanelState));

	resizeFilm();

	if (settings.film.load)
		film.load(film.getWidth(), film.getHeight(), settings.film.loadFileName, renderer.type);
	else if (settings.film.loadDir)
		film.loadMultiple(film.getWidth(), film.getHeight(), settings.film.loadDirName, renderer.type);
}

void WindowRunnerRenderState::shutdown()
{
	film.shutdown();
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

	// RENDERER / CAMERA / INTEGRATOR / FILTER / TONEMAPPER //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
		{
			if (renderer.type == RendererType::CPU)
				renderer.type = RendererType::CUDA;
			else if (renderer.type == RendererType::CUDA)
				renderer.type = RendererType::CPU;

			film.clear(renderer.type);
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.type = CameraType::ORTHOGRAPHIC;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.type = CameraType::FISHEYE;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.type = CameraType::PERSPECTIVE;

			film.clear(renderer.type);
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			if (scene.integrator.type == IntegratorType::PATH)
				scene.integrator.type = IntegratorType::DOT;
			else if (scene.integrator.type == IntegratorType::DOT)
				scene.integrator.type = IntegratorType::AMBIENT_OCCLUSION;
			else if (scene.integrator.type == IntegratorType::AMBIENT_OCCLUSION)
				scene.integrator.type = IntegratorType::DIRECT_LIGHT;
			else if (scene.integrator.type == IntegratorType::DIRECT_LIGHT)
				scene.integrator.type = IntegratorType::PATH;

			film.clear(renderer.type);
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F5))
		{
			if (scene.renderer.filter.type == FilterType::BOX)
				scene.renderer.filter.type = FilterType::TENT;
			else if (scene.renderer.filter.type == FilterType::TENT)
				scene.renderer.filter.type = FilterType::BELL;
			else if (scene.renderer.filter.type == FilterType::BELL)
				scene.renderer.filter.type = FilterType::GAUSSIAN;
			else if (scene.renderer.filter.type == FilterType::GAUSSIAN)
				scene.renderer.filter.type = FilterType::MITCHELL;
			else if (scene.renderer.filter.type == FilterType::MITCHELL)
				scene.renderer.filter.type = FilterType::LANCZOS_SINC;
			else if (scene.renderer.filter.type == FilterType::LANCZOS_SINC)
				scene.renderer.filter.type = FilterType::BOX;

			film.clear(renderer.type);
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F6))
		{
			if (scene.tonemapper.type == TonemapperType::PASSTHROUGH)
				scene.tonemapper.type = TonemapperType::LINEAR;
			else if (scene.tonemapper.type == TonemapperType::LINEAR)
				scene.tonemapper.type = TonemapperType::SIMPLE;
			else if (scene.tonemapper.type == TonemapperType::SIMPLE)
				scene.tonemapper.type = TonemapperType::REINHARD;
			else if (scene.tonemapper.type == TonemapperType::REINHARD)
				scene.tonemapper.type = TonemapperType::PASSTHROUGH;
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.fov -= 50.0f * timeStep;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.orthoSize -= 10.0f * timeStep;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.fishEyeAngle -= 50.0f * timeStep;

			film.clear(renderer.type);
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			if (scene.camera.type == CameraType::PERSPECTIVE)
				scene.camera.fov += 50.0f * timeStep;
			else if (scene.camera.type == CameraType::ORTHOGRAPHIC)
				scene.camera.orthoSize += 10.0f * timeStep;
			else if (scene.camera.type == CameraType::FISHEYE)
				scene.camera.fishEyeAngle += 50.0f * timeStep;

			film.clear(renderer.type);
		}
	}

	// RENDER SCALE //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F7))
		{
			float newScale = settings.window.renderScale * 0.5f;
			uint32_t newWidth = uint32_t(float(windowRunner.getWindowWidth()) * newScale + 0.5f);
			uint32_t newHeight = uint32_t(float(windowRunner.getWindowHeight()) * newScale + 0.5f);

			if (newWidth >= 2 && newHeight >= 2)
			{
				settings.window.renderScale = newScale;
				resizeFilm();
			}
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F8))
		{
			if (settings.window.renderScale < 1.0f)
			{
				settings.window.renderScale *= 2.0f;

				if (settings.window.renderScale > 1.0f)
					settings.window.renderScale = 1.0f;

				resizeFilm();
			}
		}
	}

	// MISC //
	
	if (windowRunner.keyWasPressed(GLFW_KEY_R))
	{
		scene.camera.reset();
		film.clear(renderer.type);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F))
	{
		scene.renderer.filtering = !scene.renderer.filtering;
		film.clear(renderer.type);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_P))
		scene.camera.enableMovement = !scene.camera.enableMovement;

	if (windowRunner.keyWasPressed(GLFW_KEY_M))
	{
		scene.general.normalMapping = !scene.general.normalMapping;
		film.clear(renderer.type);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
	{
		scene.general.normalInterpolation = !scene.general.normalInterpolation;
		film.clear(renderer.type);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_B))
	{
		scene.general.normalVisualization = !scene.general.normalVisualization;
		scene.general.interpolationVisualization = false;
		film.clear(renderer.type);
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_V))
	{
		scene.general.interpolationVisualization = !scene.general.interpolationVisualization;
		scene.general.normalVisualization = false;
		film.clear(renderer.type);
	}

	// EXPOSURE & KEY //

	if (ctrlIsPressed)
	{
		if(scene.tonemapper.type == TonemapperType::LINEAR)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene.tonemapper.linearTonemapper.exposure -= 2.0f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene.tonemapper.linearTonemapper.exposure += 2.0f * timeStep;
		}
		else if (scene.tonemapper.type == TonemapperType::SIMPLE)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene.tonemapper.simpleTonemapper.exposure -= 2.0f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene.tonemapper.simpleTonemapper.exposure += 2.0f * timeStep;
		}
		else if (scene.tonemapper.type == TonemapperType::REINHARD)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene.tonemapper.reinhardTonemapper.key -= 0.1f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene.tonemapper.reinhardTonemapper.key += 0.1f * timeStep;

			scene.tonemapper.reinhardTonemapper.key = MAX(0.0f, scene.tonemapper.reinhardTonemapper.key);
		}
	}

	// SCENE SAVING //

	if (ctrlIsPressed)
	{
		//if (windowRunner.keyWasPressed(GLFW_KEY_F1))
		//	scene.save("scene.xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
			scene.camera.saveState("camera.txt");

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			film.normalize(renderer.type);
			film.tonemap(scene.tonemapper, renderer.type);
			film.getTonemappedImage().download();
			film.getTonemappedImage().save("image.png");
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			film.normalize(renderer.type);
			film.tonemap(scene.tonemapper, renderer.type);
			film.getTonemappedImage().download();
			film.getTonemappedImage().save("image.hdr");
		}
	}

	// TEST SCENE LOADING //

	int32_t testSceneIndex = -1;

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
		film.clear(renderer.type);
	}

	scene.camera.update(timeStep);
}

void WindowRunnerRenderState::render(float timeStep, float interpolation)
{
	(void)timeStep;
	(void)interpolation;

	if (scene.camera.isMoving())
		film.clear(renderer.type);

	renderer.filtering = !film.hasBeenCleared();
	film.resetCleared();

	RenderJob job;
	job.scene = &scene;
	job.film = &film;
	job.interrupted = false;
	job.totalSampleCount = 0;
	
	renderer.render(job);
	film.normalize(renderer.type);
	film.tonemap(scene.tonemapper, renderer.type);
	film.updateTexture(renderer.type);
	filmQuad.render(film);
	infoPanel.render(renderer, job);
}

void WindowRunnerRenderState::windowResized(uint32_t width, uint32_t height)
{
	(void)width;
	(void)height;

	resizeFilm();
}

void WindowRunnerRenderState::resizeFilm()
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();

	uint32_t filmWidth = uint32_t(float(windowRunner.getWindowWidth()) * settings.window.renderScale + 0.5);
	uint32_t filmHeight = uint32_t(float(windowRunner.getWindowHeight()) * settings.window.renderScale + 0.5);

    filmWidth = MAX(uint32_t(1), filmWidth);
    filmHeight = MAX(uint32_t(1), filmHeight);

	film.resize(filmWidth, filmHeight, renderer.type);
	renderer.resize(filmWidth, filmHeight);
	scene.camera.setImagePlaneSize(filmWidth, filmHeight);
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/App.h"
#include "Core/Camera.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Runners/WindowRunner.h"
#include "Runners/WindowRunnerRenderState.h"
#include "TestScenes/TestScene.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"

using namespace Raycer;

WindowRunnerRenderState::WindowRunnerRenderState()
{
	scene = new Scene();
	film = new Film();
}

WindowRunnerRenderState::~WindowRunnerRenderState()
{
	delete scene;
	delete film;
}

void WindowRunnerRenderState::initialize()
{
	Settings& settings = App::getSettings();
	
	if (settings.scene.useTestScene)
		*scene = TestScene::create(settings.scene.testSceneNumber);
	else
		*scene = Scene::load(settings.scene.fileName);

	scene->initialize();
	renderer.initialize();
	filmQuad.initialize();
	infoPanel.initialize();
	infoPanel.setState(InfoPanelState(settings.window.infoPanelState));

	resizeFilm();
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

	// RENDERER / CAMERA / INTEGRATOR / FILTER / TONEMAPPER //

	if (!ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
		{
			if (renderer.type == RendererType::CPU)
				renderer.type = RendererType::CUDA;
			else if (renderer.type == RendererType::CUDA)
				renderer.type = RendererType::CPU;

			film->clear();
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
		{
			if (scene->camera.type == CameraType::PERSPECTIVE)
				scene->camera.type = CameraType::ORTHOGRAPHIC;
			else if (scene->camera.type == CameraType::ORTHOGRAPHIC)
				scene->camera.type = CameraType::FISHEYE;
			else if (scene->camera.type == CameraType::FISHEYE)
				scene->camera.type = CameraType::PERSPECTIVE;

			film->clear();
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			if (scene->integrator.type == IntegratorType::DOT)
				scene->integrator.type = IntegratorType::PATH;
			else if (scene->integrator.type == IntegratorType::PATH)
				scene->integrator.type = IntegratorType::DOT;

			film->clear();
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F5))
		{
			if (scene->filter.type == FilterType::BOX)
				scene->filter.type = FilterType::TENT;
			else if (scene->filter.type == FilterType::TENT)
				scene->filter.type = FilterType::BELL;
			else if (scene->filter.type == FilterType::BELL)
				scene->filter.type = FilterType::GAUSSIAN;
			else if (scene->filter.type == FilterType::GAUSSIAN)
				scene->filter.type = FilterType::MITCHELL;
			else if (scene->filter.type == FilterType::MITCHELL)
				scene->filter.type = FilterType::LANCZOS_SINC;
			else if (scene->filter.type == FilterType::LANCZOS_SINC)
				scene->filter.type = FilterType::BOX;

			film->clear();
		}

		if (windowRunner.keyWasPressed(GLFW_KEY_F6))
		{
			if (scene->tonemapper.type == TonemapperType::PASSTHROUGH)
				scene->tonemapper.type = TonemapperType::LINEAR;
			else if (scene->tonemapper.type == TonemapperType::LINEAR)
				scene->tonemapper.type = TonemapperType::SIMPLE;
			else if (scene->tonemapper.type == TonemapperType::SIMPLE)
				scene->tonemapper.type = TonemapperType::REINHARD;
			else if (scene->tonemapper.type == TonemapperType::REINHARD)
				scene->tonemapper.type = TonemapperType::PASSTHROUGH;
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			if (scene->camera.type == CameraType::PERSPECTIVE)
				scene->camera.fov -= 50.0f * timeStep;
			else if (scene->camera.type == CameraType::ORTHOGRAPHIC)
				scene->camera.orthoSize -= 10.0f * timeStep;
			else if (scene->camera.type == CameraType::FISHEYE)
				scene->camera.fishEyeAngle -= 50.0f * timeStep;

			film->clear();
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			if (scene->camera.type == CameraType::PERSPECTIVE)
				scene->camera.fov += 50.0f * timeStep;
			else if (scene->camera.type == CameraType::ORTHOGRAPHIC)
				scene->camera.orthoSize += 10.0f * timeStep;
			else if (scene->camera.type == CameraType::FISHEYE)
				scene->camera.fishEyeAngle += 50.0f * timeStep;

			film->clear();
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
		scene->camera.reset();
		film->clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_F))
	{
		scene->general.pixelFiltering = !scene->general.pixelFiltering;
		film->clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_P))
		scene->camera.enableMovement = !scene->camera.enableMovement;

	if (windowRunner.keyWasPressed(GLFW_KEY_M))
	{
		scene->general.normalMapping = !scene->general.normalMapping;
		film->clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_N))
	{
		scene->general.normalInterpolation = !scene->general.normalInterpolation;
		film->clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_B))
	{
		scene->general.normalVisualization = !scene->general.normalVisualization;
		scene->general.interpolationVisualization = false;
		film->clear();
	}

	if (windowRunner.keyWasPressed(GLFW_KEY_V))
	{
		scene->general.interpolationVisualization = !scene->general.interpolationVisualization;
		scene->general.normalVisualization = false;
		film->clear();
	}

	// EXPOSURE & KEY //

	if (ctrlIsPressed)
	{
		if(scene->tonemapper.type == TonemapperType::LINEAR)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene->tonemapper.linearTonemapper.exposure -= 2.0f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene->tonemapper.linearTonemapper.exposure += 2.0f * timeStep;
		}
		else if (scene->tonemapper.type == TonemapperType::SIMPLE)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene->tonemapper.simpleTonemapper.exposure -= 2.0f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene->tonemapper.simpleTonemapper.exposure += 2.0f * timeStep;
		}
		else if (scene->tonemapper.type == TonemapperType::REINHARD)
		{
			if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
				scene->tonemapper.reinhardTonemapper.key -= 0.1f * timeStep;
			else if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
				scene->tonemapper.reinhardTonemapper.key += 0.1f * timeStep;

			scene->tonemapper.reinhardTonemapper.key = std::max(0.0f, scene->tonemapper.reinhardTonemapper.key);
		}
	}

	// SCENE SAVING //

	if (ctrlIsPressed)
	{
		if (windowRunner.keyWasPressed(GLFW_KEY_F1))
			scene->save("scene->xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_F2))
			renderer.save("renderer.xml");

		if (windowRunner.keyWasPressed(GLFW_KEY_F3))
			scene->camera.saveState("camera.txt");

		if (windowRunner.keyWasPressed(GLFW_KEY_F4))
		{
			film->generateImage(scene->tonemapper);
			film->getImage().save("image.png");
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

		delete scene;
		scene = new Scene();

		try
		{
			*scene = TestScene::create(testSceneIndex);
			scene->initialize();
		}
		catch (const std::exception& ex)
		{
			log.logWarning("Could not create test scene: %s", ex.what());

			*scene = Scene();
			scene->initialize();
		}

		scene->camera.setImagePlaneSize(film->getWidth(), film->getHeight());
		film->clear();
	}

	scene->camera.update(timeStep);
}

void WindowRunnerRenderState::render(float timeStep, float interpolation)
{
	(void)timeStep;
	(void)interpolation;

	if (scene->camera.isMoving())
		film->clear();

	RenderJob job;
	job.scene = scene;
	job.film = film;
	job.interrupted = false;
	job.sampleCount = 0;
	
	renderer.render(job);
	film->generateImage(scene->tonemapper);
	filmQuad.upload(*film);
	filmQuad.render();
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

    filmWidth = std::max(uint32_t(1), filmWidth);
    filmHeight = std::max(uint32_t(1), filmHeight);

	film->resize(filmWidth, filmHeight);
	filmQuad.resize(filmWidth, filmHeight);

	scene->camera.setImagePlaneSize(filmWidth, filmHeight);
}

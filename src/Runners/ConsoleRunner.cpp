// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "App.h"
#include "Core/Camera.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Runners/ConsoleRunner.h"
#include "TestScenes/TestScene.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "Utils/SysUtils.h"

using namespace Raycer;
using namespace std::chrono;

int ConsoleRunner::run()
{
	Settings& settings = App::getSettings();
	Log& log = App::getLog();

	samplesPerSecondAverage.setAlpha(0.05f);

	Timer totalElapsedTimer;
	Renderer renderer;

	Scene scene = TestScene::create(settings.scene.testSceneNumber);
	Film film(false);

	scene.initialize();
	film.initialize();
	renderer.initialize(settings);

	film.resize(settings.image.width, settings.image.height);
	film.clear(renderer.type);

	scene.camera.setImagePlaneSize(settings.image.width, settings.image.height);
	scene.camera.update(0.0f);

	renderJob.scene = &scene;
	renderJob.film = &film;
	renderJob.interrupted = false;
	renderJob.sampleCount = 0;
	
	SysUtils::setConsoleTextColor(ConsoleTextColor::WHITE_ON_BLACK);

	uint64_t totalSamples = uint64_t(settings.image.width) * uint64_t(settings.image.height) * uint64_t(scene.renderer.pixelSamples);

	std::cout << tfm::format("\nRendering started (size: %dx%d, pixels: %s, samples: %s, pixel samples: %d)\n\n",
		settings.image.width,
		settings.image.height,
		StringUtils::humanizeNumber(double(settings.image.width * settings.image.height)),
		StringUtils::humanizeNumber(double(totalSamples)),
		scene.renderer.pixelSamples);

	Timer renderingElapsedTimer;
	renderingElapsedTimer.setAveragingAlpha(0.05f);
	renderingElapsedTimer.setTargetValue(float(totalSamples));

	std::atomic<bool> renderThreadFinished(false);
	std::exception_ptr renderThreadException = nullptr;

	auto renderThreadFunction = [&]()
	{
		try
		{
			renderer.render(renderJob);
		}
		catch (...)
		{
			renderThreadException = std::current_exception();
		}

		renderThreadFinished = true;
	};

	std::thread renderThread(renderThreadFunction);

	while (!renderThreadFinished)
	{
		renderingElapsedTimer.updateCurrentValue(float(renderJob.sampleCount));

		auto elapsed = renderingElapsedTimer.getElapsed();
		auto remaining = renderingElapsedTimer.getRemaining();

		if (elapsed.totalMilliseconds > 0)
			samplesPerSecondAverage.addMeasurement(float(renderJob.sampleCount) / (float(elapsed.totalMilliseconds) / 1000.0f));

		printProgress(renderingElapsedTimer.getPercentage(), elapsed, remaining, film.pixelSamples);
		std::this_thread::sleep_for(std::chrono::milliseconds(250));
	}

	renderThread.join();

	if (renderThreadException != nullptr)
		std::rethrow_exception(renderThreadException);

	renderingElapsedTimer.updateCurrentValue(float(renderJob.sampleCount));

	auto elapsed = renderingElapsedTimer.getElapsed();
	auto remaining = renderingElapsedTimer.getRemaining();

	printProgress(renderingElapsedTimer.getPercentage(), elapsed, remaining, film.pixelSamples);

	float totalSamplesPerSecond = 0.0f;

	if (elapsed.totalMilliseconds > 0)
		totalSamplesPerSecond = float(renderJob.sampleCount) / (float(elapsed.totalMilliseconds) / 1000.0f);

	std::cout << tfm::format("\n\nRendering %s (time: %s, samples/s: %s)\n\n",
		renderJob.interrupted ? "interrupted" : "finished",
		elapsed.getString(true),
		StringUtils::humanizeNumber(totalSamplesPerSecond));

	SysUtils::setConsoleTextColor(ConsoleTextColor::DEFAULT);

	film.normalize(renderer.type);
	film.tonemap(scene.tonemapper, renderer.type);
	film.getTonemappedImage().download();

	log.logInfo("Total elapsed time: %s", totalElapsedTimer.getElapsed().getString(true));

	if (!renderJob.interrupted)
	{
		film.getTonemappedImage().save(settings.image.fileName);

		if (settings.image.autoView)
			SysUtils::openFileExternally(settings.image.fileName);
	}
	else
		film.getTonemappedImage().save("partial_image.png");

	return 0;
}

void ConsoleRunner::interrupt()
{
	renderJob.interrupted = true;
}

void ConsoleRunner::printProgress(float percentage, const TimerData& elapsed, const TimerData& remaining, uint32_t pixelSamples)
{
	uint32_t barCount = uint32_t(percentage + 0.5f) / 4;

    tfm::printf("[");

	for (uint32_t i = 0; i < barCount; ++i)
        tfm::printf("=");

	if (barCount < 25)
	{
        tfm::printf(">");

		for (uint32_t i = 0; i < (24 - barCount); ++i)
            tfm::printf(" ");
	}

    tfm::printf("] ");
    tfm::printf("%.2f %% | ", percentage);
    tfm::printf("Elapsed: %s | ", elapsed.getString());
    tfm::printf("Remaining: %s | ", remaining.getString());
	tfm::printf("Samples/s: %s | ", StringUtils::humanizeNumber(samplesPerSecondAverage.getAverage()));
	tfm::printf("Pixel samples: %d", pixelSamples);
    tfm::printf("          \r");
}

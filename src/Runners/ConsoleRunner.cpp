// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Runners/ConsoleRunner.h"
#include "App.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "Utils/SysUtils.h"
#include "Rendering/Film.h"
#include "Scenes/Scene.h"
#include "Tracers/Tracer.h"
#include "Tracers/TracerState.h"
#include "Tracing/Camera.h"

using namespace Raycer;
using namespace std::chrono;

int ConsoleRunner::run()
{
	Settings& settings = App::getSettings();
	Log& log = App::getLog();

	Timer totalElapsedTimer;

	interrupted = false;

	samplesPerSecondAverage.setAlpha(0.05);
	pixelsPerSecondAverage.setAlpha(0.05);
	raysPerSecondAverage.setAlpha(0.05);
	pathsPerSecondAverage.setAlpha(0.05);

	Scene scene;

	if (settings.scene.enableTestScenes)
		scene = Scene::createTestScene(settings.scene.testSceneNumber);
	else
		scene = Scene::loadFromFile(settings.scene.fileName);

	scene.initialize();
	scene.camera.setImagePlaneSize(settings.image.width, settings.image.height);
	scene.camera.update(0.0);

	Film film;
	film.resize(settings.image.width, settings.image.height);

	if (settings.film.restoreFromFile)
		film.load(settings.film.restoreFileName);

	TracerState state;
	state.scene = &scene;
	state.film = &film;
	state.filmWidth = settings.image.width;
	state.filmHeight = settings.image.height;
	state.filmPixelOffset = 0;
	state.filmPixelCount = state.filmWidth * state.filmHeight;

	run(state);

	log.logInfo("Total elapsed time: %s", totalElapsedTimer.getElapsed().getString(true));

	if (!interrupted)
	{
		film.getOutputImage().save(settings.image.fileName);

		if (settings.image.autoView)
			SysUtils::openFileExternally(settings.image.fileName);
	}
	else
		film.getOutputImage().save("partial_result.png");

	return 0;
}

void ConsoleRunner::run(TracerState& state)
{
	interrupted = false;

	SysUtils::setConsoleTextColor(ConsoleTextColor::WHITE_ON_BLACK);

	Settings& settings = App::getSettings();
	Scene& scene = *state.scene;
	auto tracer = Tracer::getTracer(state.scene->general.tracerType);
	
	uint64_t totalSamples =
		state.filmPixelCount *
		tracer->getPixelSampleCount(scene) *
		tracer->getSamplesPerPixel(scene);

	std::cout << tfm::format("\nTracing started (threads: %d, dimensions: %dx%d, offset: %d, pixels: %d, samples: %s, pixel samples: %d)\n\n",
		settings.general.maxThreadCount,
		state.filmWidth,
		state.filmHeight,
		state.filmPixelOffset,
		state.filmPixelCount,
		StringUtils::humanizeNumber(double(totalSamples)),
		tracer->getPixelSampleCount(scene)
		);

	Timer tracingElapsedTimer;
	tracingElapsedTimer.setAveragingAlpha(0.05);
	tracingElapsedTimer.setTargetValue(double(totalSamples));

	std::atomic<bool> renderThreadFinished(false);
	std::exception_ptr renderThreadException = nullptr;

	auto renderThreadFunction = [&]()
	{
		try
		{
			tracer->run(state, interrupted);
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
		tracingElapsedTimer.updateCurrentValue(double(state.sampleCount));

		auto elapsed = tracingElapsedTimer.getElapsed();
		auto remaining = tracingElapsedTimer.getRemaining();

		if (elapsed.totalMilliseconds > 0)
		{
			samplesPerSecondAverage.addMeasurement(double(state.sampleCount) / (double(elapsed.totalMilliseconds) / 1000.0));
			pixelsPerSecondAverage.addMeasurement(double(state.pixelCount) / (double(elapsed.totalMilliseconds) / 1000.0));
			raysPerSecondAverage.addMeasurement(double(state.rayCount) / (double(elapsed.totalMilliseconds) / 1000.0));
			pathsPerSecondAverage.addMeasurement(double(state.pathCount) / (double(elapsed.totalMilliseconds) / 1000.0));
		}

		printProgress(tracingElapsedTimer.getPercentage(), elapsed, remaining, state.pixelSampleCount);
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}

	renderThread.join();

	if (renderThreadException != nullptr)
		std::rethrow_exception(renderThreadException);

	tracingElapsedTimer.updateCurrentValue(double(state.sampleCount));

	auto elapsed = tracingElapsedTimer.getElapsed();
	auto remaining = tracingElapsedTimer.getRemaining();

	printProgress(tracingElapsedTimer.getPercentage(), elapsed, remaining, state.pixelSampleCount);

	double totalSamplesPerSecond = 0.0;
	double totalPixelsPerSecond = 0.0;
	double totalRaysPerSecond = 0.0;
	double totalPathsPerSecond = 0.0;

	if (elapsed.totalMilliseconds > 0)
	{
		totalSamplesPerSecond = double(state.sampleCount) / (double(elapsed.totalMilliseconds) / 1000.0);
		totalPixelsPerSecond = double(state.pixelCount) / (double(elapsed.totalMilliseconds) / 1000.0);
		totalRaysPerSecond = double(state.rayCount) / (double(elapsed.totalMilliseconds) / 1000.0);
		totalPathsPerSecond = double(state.pathCount) / (double(elapsed.totalMilliseconds) / 1000.0);
	}

	std::cout << tfm::format("\n\nTracing %s (time: %s, samples/s: %s, pixels/s: %s, rays/s: %s, paths/s: %s)\n\n",
		interrupted ? "interrupted" : "finished",
		elapsed.getString(true),
		StringUtils::humanizeNumber(totalSamplesPerSecond),
		StringUtils::humanizeNumber(totalPixelsPerSecond),
		StringUtils::humanizeNumber(totalRaysPerSecond),
		StringUtils::humanizeNumber(totalPathsPerSecond));

	SysUtils::setConsoleTextColor(ConsoleTextColor::DEFAULT);

	state.film->generateOutputImage(*state.scene);
}

void ConsoleRunner::interrupt()
{
	interrupted = true;
}

void ConsoleRunner::printProgress(double percentage_, const TimerData& elapsed, const TimerData& remaining, uint64_t pixelSamples)
{
	uint64_t percentage = uint64_t(percentage_ + 0.5);
	uint64_t barCount = percentage / 4;

    tfm::printf("[");

	for (uint64_t i = 0; i < barCount; ++i)
        tfm::printf("=");

	if (barCount < 25)
	{
        tfm::printf(">");

		for (uint64_t i = 0; i < (24 - barCount); ++i)
            tfm::printf(" ");
	}

    tfm::printf("] ");
    tfm::printf("%d %% | ", percentage);
    tfm::printf("Elapsed: %s | ", elapsed.getString());
    tfm::printf("Remaining: %s | ", remaining.getString());
	tfm::printf("Samples/s: %s | ", StringUtils::humanizeNumber(samplesPerSecondAverage.getAverage()));
	tfm::printf("Pixels/s: %s | ", StringUtils::humanizeNumber(pixelsPerSecondAverage.getAverage()));
	tfm::printf("Rays/s: %s | ", StringUtils::humanizeNumber(raysPerSecondAverage.getAverage()));
	tfm::printf("Paths/s: %s | ", StringUtils::humanizeNumber(pathsPerSecondAverage.getAverage()));
	tfm::printf("Pixel samples: %d", pixelSamples);
    tfm::printf("          \r");
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

#include "tinyformat/tinyformat.h"

#include "Runners/ConsoleRunner.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Utils/OpenCL.h"
#include "Utils/Image.h"
#include "Raytracing/Scene.h"
#include "Raytracing/Raytracer.h"
#include "CLRaytracing/CLRaytracer.h"
#include "CLRaytracing/CLStructs.h"

using namespace Raycer;
using namespace std::chrono;

int ConsoleRunner::run()
{
	Log& log = App::getLog();
	Settings& settings = App::getSettings();

	interrupted = false;

	pixelsPerSecondAverage.setAlpha(0.1);
	pixelsPerSecondAverage.setAverage(1.0);
	raysPerSecondAverage.setAlpha(0.1);
	raysPerSecondAverage.setAverage(1.0);
	remainingTimeAverage.setAlpha(0.1);
	remainingTimeAverage.setAverage(0.0);

	Scene scene;
	//scene.loadFromFile(settings.scene.fileName);
	scene = Scene::createTestScene();
	scene.initialize();
	scene.camera.setImagePlaneSize(settings.image.width, settings.image.height);
	scene.camera.precalculate();

	RaytracerConfig config;
	config.renderTarget = &resultImage;
	config.scene = &scene;
	config.sceneWidth = settings.image.width;
	config.sceneHeight = settings.image.height;
	config.pixelOffset = 0;
	config.pixelCount = config.sceneWidth * config.sceneHeight;
	config.isInteractive = false;

	resultImage.setSize(config.sceneWidth, config.sceneHeight);

	run(config);

	if (!interrupted)
	{
		resultImage.flip();
		resultImage.swapBytes();
		resultImage.saveAs(settings.image.fileName);

		if (settings.image.autoView)
		{
			log.logInfo("Opening the image in an external viewer");
#ifdef _WIN32
			ShellExecuteA(NULL, "open", settings.image.fileName.c_str(), NULL, NULL, SW_SHOWNORMAL);
#endif
		}
	}

	return 0;
}

void ConsoleRunner::run(RaytracerConfig& config)
{
	Settings& settings = App::getSettings();
	OpenCL& openCL = App::getOpenCL();
	Raytracer& raytracer = App::getRaytracer();
	CLRaytracer& gpuRaytracer = App::getGpuRaytracer();

	interrupted = false;

	if (settings.openCL.enabled && !openCLInitialized)
	{
		openCL.initialize();
		openCL.loadKernels();

		openCLInitialized = true;
	}

	if (settings.openCL.enabled)
	{
		gpuRaytracer.initialize();
		gpuRaytracer.resizePixelBuffer(config.sceneWidth, config.sceneHeight);
		gpuRaytracer.readScene(*config.scene);
		gpuRaytracer.uploadData();
	}
	
	std::atomic<bool> finished;

	auto renderFunction = [&]()
	{
		if (!settings.openCL.enabled)
			raytracer.trace(config, interrupted);
		else
			gpuRaytracer.trace(interrupted);

		finished = true;
	};

	std::cout << tfm::format("\nStart raytracing (dimensions: %dx%d, pixels: %s, size: %s, offset: %d)\n\n", config.sceneWidth, config.sceneHeight, humanizeNumberDecimal(config.pixelCount), humanizeNumberBytes(config.pixelCount * 4 * 4), config.pixelOffset);

	auto startTime = high_resolution_clock::now();
	std::thread renderThread(renderFunction);

	while (!finished)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(100));

		if (!settings.openCL.enabled)
			printProgress(startTime, config.pixelCount, config.pixelsProcessed, config.raysProcessed);
		else
			printProgressOpenCL(startTime);
	}

	renderThread.join();

	if (!settings.openCL.enabled)
		printProgress(startTime, config.pixelCount, config.pixelsProcessed, config.raysProcessed);
	else
		printProgressOpenCL(startTime);

	if (settings.openCL.enabled)
		config.pixelsProcessed = config.pixelCount;

	auto elapsedTime = high_resolution_clock::now() - startTime;
	int totalElapsedSeconds = (int)duration_cast<std::chrono::seconds>(elapsedTime).count();
	int totalElapsedMilliseconds = (int)duration_cast<std::chrono::milliseconds>(elapsedTime).count();
	int elapsedHours = totalElapsedSeconds / 3600;
	int elapsedMinutes = (totalElapsedSeconds - elapsedHours * 3600) / 60;
	int elapsedSeconds = totalElapsedSeconds - elapsedHours * 3600 - elapsedMinutes * 60;
	int elapsedMilliseconds = totalElapsedMilliseconds - totalElapsedSeconds * 1000;

	double totalPixelsPerSecond = 0.0;
	double totalRaysPerSecond = 0.0;

	if (totalElapsedMilliseconds > 0)
	{
		totalPixelsPerSecond = (double)config.pixelsProcessed / ((double)totalElapsedMilliseconds / 1000.0);
		totalRaysPerSecond = (double)config.raysProcessed / ((double)totalElapsedMilliseconds / 1000.0);
	}
	
	std::string timeString = tfm::format("%02d:%02d:%02d.%03d", elapsedHours, elapsedMinutes, elapsedSeconds, elapsedMilliseconds);

	if (!settings.openCL.enabled)
		std::cout << tfm::format("\n\nRaytracing %s (time: %s, pixels: %s, pixels/s: %s, rays: %s, rays/s: %s)\n\n", interrupted ? "interrupted" : "finished", timeString, humanizeNumberDecimal(config.pixelsProcessed.load()), humanizeNumberDecimal(totalPixelsPerSecond), humanizeNumberDecimal(config.raysProcessed.load()), humanizeNumberDecimal(totalRaysPerSecond));
	else
		std::cout << tfm::format("\n\nRaytracing %s (time: %s, pixels: %s, pixels/s: %s)\n\n", interrupted ? "interrupted" : "finished", timeString, humanizeNumberDecimal(config.pixelsProcessed.load()), humanizeNumberDecimal(totalPixelsPerSecond));

	if (!interrupted && settings.openCL.enabled)
	{
		gpuRaytracer.downloadImage();
		resultImage = gpuRaytracer.getImage();
	}
}

void ConsoleRunner::interrupt()
{
	interrupted = true;
}

Image& ConsoleRunner::getResultImage()
{
	return resultImage;
}

void ConsoleRunner::printProgress(const time_point<system_clock>& startTime, int totalPixelCount, int pixelsProcessed, int raysProcessed)
{
	auto elapsedTime = high_resolution_clock::now() - startTime;
	double elapsedSeconds = (double)duration_cast<std::chrono::milliseconds>(elapsedTime).count() / 1000.0;
	double msPerPixel = 0.0;

	if (pixelsProcessed > 0)
		msPerPixel = (double)duration_cast<std::chrono::milliseconds>(elapsedTime).count() / (double)pixelsProcessed;

	auto estimatedTime = std::chrono::milliseconds((int)(msPerPixel * (double)totalPixelCount + 0.5));
	auto remainingTime = estimatedTime - elapsedTime;

	int percentage = (int)(((double)pixelsProcessed / (double)totalPixelCount) * 100.0 + 0.5);
	int barCount = percentage / 4;

	printf("[");

	for (int i = 0; i < barCount; ++i)
		printf("=");

	if (barCount < 25)
	{
		printf(">");

		for (int i = 0; i < (24 - barCount); ++i)
			printf(" ");
	}

	if (elapsedSeconds > 0.0)
	{
		pixelsPerSecondAverage.addMeasurement((double)pixelsProcessed / elapsedSeconds);
		raysPerSecondAverage.addMeasurement((double)raysProcessed / elapsedSeconds);
	}
	
	remainingTimeAverage.addMeasurement((double)duration_cast<std::chrono::seconds>(remainingTime).count());

	if (pixelsProcessed == totalPixelCount)
		remainingTimeAverage.setAverage(0.0);

	int totalRemainingSeconds = (int)(remainingTimeAverage.getAverage() + 0.5);
	int remainingHours = totalRemainingSeconds / 3600;
	int remainingMinutes = (totalRemainingSeconds - remainingHours * 3600) / 60;
	int remainingSeconds = totalRemainingSeconds - remainingHours * 3600 - remainingMinutes * 60;

	printf("] ");
	printf("%d %% | ", percentage);
	printf("Remaining time: %02d:%02d:%02d | ", remainingHours, remainingMinutes, remainingSeconds);
	printf("Pixels/s: %s | ", humanizeNumberDecimal(pixelsPerSecondAverage.getAverage()).c_str());
	printf("Rays/s: %s", humanizeNumberDecimal(raysPerSecondAverage.getAverage()).c_str());
	printf("          \r");
}

void ConsoleRunner::printProgressOpenCL(const std::chrono::time_point<std::chrono::system_clock>& startTime)
{
	auto elapsedTime = high_resolution_clock::now() - startTime;
	int totalElapsedSeconds = (int)duration_cast<std::chrono::seconds>(elapsedTime).count();
	int elapsedHours = totalElapsedSeconds / 3600;
	int elapsedMinutes = (totalElapsedSeconds - elapsedHours * 3600) / 60;
	int elapsedSeconds = totalElapsedSeconds - elapsedHours * 3600 - elapsedMinutes * 60;

	if (++openCLProgressCounter1 % 5 == 0)
		++openCLProgressCounter2;

	char progressChar;

	switch (openCLProgressCounter2 % 4)
	{
		case 1: progressChar = '\\'; break;
		case 2: progressChar = '|'; break;
		case 3: progressChar = '/'; break;
		default: progressChar = '-'; break;
	}

	printf("[%c] ", progressChar);
	printf("Elapsed time: %02d:%02d:%02d", elapsedHours, elapsedMinutes, elapsedSeconds);
	printf("          \r");
}

std::string ConsoleRunner::humanizeNumberDecimal(double value)
{
	const char* prefixes[] = {"", "k", "M", "G", "T", "P", "E", "Z", "Y"};

	for (int i = 0; i < 9; ++i)
	{
		if (value < 1000.0)
			return tfm::format("%.2f %s", value, prefixes[i]);
		else
			value /= 1000.0;
	}

	return tfm::format("%.2f %s", value, "Y");
}

std::string ConsoleRunner::humanizeNumberBytes(double value)
{
	const char* prefixes[] = { "B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };

	for (int i = 0; i < 9; ++i)
	{
		if (value < 1024.0)
			return tfm::format("%.2f %s", value, prefixes[i]);
		else
			value /= 1024.0;
	}

	return tfm::format("%.2f %s", value, "YB");
}

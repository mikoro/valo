// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef _WIN32
#include <windows.h>
#endif

#ifdef __linux
#include <signal.h>
#endif

#ifdef RUN_UNIT_TESTS
#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "App.h"
#include "Core/Common.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Utils/CudaUtils.h"
#include "Runners/WindowRunner.h"
#include "Runners/ConsoleRunner.h"

using namespace Valo;

int main(int argc, char** argv)
{
	int result;

#ifdef RUN_UNIT_TESTS
	result = Catch::Session().run(argc, argv);
#else
	result = App::run(argc, argv);
#endif

#ifdef USE_CUDA
	cudaDeviceReset();
#endif

	return result;
}

#ifdef _WIN32
BOOL consoleCtrlHandler(DWORD fdwCtrlType)
{
	if (fdwCtrlType == CTRL_C_EVENT)
	{
		App::getLog().logInfo("Interrupted!");
		App::getConsoleRunner().interrupt();

		return true;
	}

	return false;
}
#else
void signalHandler(int signal)
{
	(void)signal;

	App::getLog().logInfo("Interrupted!");
	App::getConsoleRunner().interrupt();
}
#endif

int App::run(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleCtrlHandler(PHANDLER_ROUTINE(consoleCtrlHandler), TRUE);
#else
	signal(SIGINT, signalHandler);
	signal(SIGTERM, signalHandler);
#endif

	Log& log = getLog();

	try
	{
		Settings& settings = getSettings();
		WindowRunner& windowRunner = getWindowRunner();
		ConsoleRunner& consoleRunner = getConsoleRunner();

		if (!settings.load(argc, argv))
			return 0;

		log.logInfo(std::string("Valo v") + VALO_VERSION);

		if (settings.general.maxCpuThreadCount == 0)
			settings.general.maxCpuThreadCount = std::thread::hardware_concurrency();

		log.logInfo("CPU thread count: %s", settings.general.maxCpuThreadCount);

#ifdef USE_CUDA

		int deviceCount;
		CudaUtils::checkError(cudaGetDeviceCount(&deviceCount), "Could not get device count");
		CudaUtils::checkError(cudaSetDevice(settings.general.cudaDeviceNumber), "Could not set device");

		log.logInfo("CUDA selected device: %d (device count: %d)", settings.general.cudaDeviceNumber, deviceCount);

		cudaDeviceProp deviceProp;
		CudaUtils::checkError(cudaGetDeviceProperties(&deviceProp, settings.general.cudaDeviceNumber), "Could not get device properties");

		int driverVersion;
		CudaUtils::checkError(cudaDriverGetVersion(&driverVersion), "Could not get driver version");

		int runtimeVersion;
		CudaUtils::checkError(cudaRuntimeGetVersion(&runtimeVersion), "Could not get runtime version");

		log.logInfo("CUDA device: %s | Compute capability: %d.%d | Driver version: %d | Runtime version: %d", deviceProp.name, deviceProp.major, deviceProp.minor, driverVersion, runtimeVersion);

#endif

		if (settings.general.windowed)
			return windowRunner.run();
		else
			return consoleRunner.run();
	}
	catch (...)
	{
		log.logException(std::current_exception());
		return -1;
	}
}

Log& App::getLog()
{
	static Log log("valo.log");
	return log;
}

Settings& App::getSettings()
{
	static Settings settings;
	return settings;
}

WindowRunner& App::getWindowRunner()
{
	static WindowRunner windowRunner;
	return windowRunner;
}

ConsoleRunner& App::getConsoleRunner()
{
	static ConsoleRunner consoleRunner;
	return consoleRunner;
}

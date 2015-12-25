// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "App.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Runners/WindowRunner.h"
#include "Runners/ConsoleRunner.h"
#include "Runners/NetworkRunner.h"

#ifdef RUN_UNIT_TESTS
#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#endif

using namespace Raycer;

int main(int argc, char** argv)
{
#ifdef RUN_UNIT_TESTS
	return Catch::Session().run(argc, argv);
#else
	return App().run(argc, argv);
#endif
}

#ifdef _WIN32
BOOL consoleCtrlHandler(DWORD fdwCtrlType)
{
	if (fdwCtrlType == CTRL_C_EVENT)
	{
		App::getConsoleRunner().interrupt();
		App::getNetworkRunner().interrupt();

		return true;
	}
	
	return false;
}
#else
void signalHandler(int signal)
{
	(void)signal;

	App::getConsoleRunner().interrupt();
	App::getNetworkRunner().interrupt();
}
#endif

int App::run(int argc, char** argv)
{
#ifdef _WIN32
	SetConsoleCtrlHandler(PHANDLER_ROUTINE(consoleCtrlHandler), TRUE);
#else
	signal(SIGINT, signalHandler);
#endif

	Log& log = getLog();

	try
	{
		Settings& settings = getSettings();
		WindowRunner& windowRunner = getWindowRunner();
		ConsoleRunner& consoleRunner = getConsoleRunner();
		NetworkRunner& networkRunner = getNetworkRunner();

		if (!settings.load(argc, argv))
			return 0;

		log.logInfo(std::string("Raycer v") + RAYCER_VERSION);

		if (settings.network.isClient && settings.network.isServer)
			throw std::runtime_error("Could not be both a server and a client at the same time");

		if (settings.network.isClient || settings.network.isServer)
			settings.general.interactive = false;

		if (settings.general.interactive)
			return windowRunner.run();

		if (settings.network.isClient || settings.network.isServer)
			return networkRunner.run();
		
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
	static Log log("raycer.log");
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

NetworkRunner& App::getNetworkRunner()
{
	static NetworkRunner networkRunner;
	return networkRunner;
}

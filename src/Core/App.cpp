// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include "Core/App.h"
#include "Core/Common.h"
#include "Utils/Settings.h"
#include "Utils/Log.h"
#include "Runners/WindowRunner.h"
#include "Runners/ConsoleRunner.h"
#include "Utils/ImagePool.h"

using namespace Raycer;

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
#endif

	Log& log = getLog();

	try
	{
		Settings& settings = getSettings();
		WindowRunner& windowRunner = getWindowRunner();
		ConsoleRunner& consoleRunner = getConsoleRunner();

		if (!settings.load(argc, argv))
			return 0;

		log.logInfo(std::string("Raycer v") + RAYCER_VERSION);

		if (settings.window.enabled)
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

ImagePool& App::getImagePool()
{
	static ImagePool imagePool;
	return imagePool;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Log;
	class Settings;
	class WindowRunner;
	class ConsoleRunner;

	class App
	{
	public:

		static int run(int argc, char** argv);

		static Log& getLog();
		static Settings& getSettings();
		static WindowRunner& getWindowRunner();
		static ConsoleRunner& getConsoleRunner();
	};
}

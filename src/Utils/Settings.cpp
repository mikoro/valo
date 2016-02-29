// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/Settings.h"

using namespace Raycer;

namespace po = boost::program_options;

bool Settings::load(int argc, char** argv)
{
	po::options_description options("Options");

	options.add_options()

		("help", "")

		("general.maxThreadCount", po::value(&general.maxThreadCount)->default_value(4), "")

		("interactive.enabled", po::value(&interactive.enabled)->default_value(true), "")
		("interactive.checkGLErrors", po::value(&interactive.checkGLErrors)->default_value(true), "")
		("interactive.renderScale", po::value(&interactive.renderScale)->default_value(0.25f), "")
		("interactive.infoPanelState", po::value(&interactive.infoPanelState)->default_value(2), "")
		("interactive.infoPanelFontSize", po::value(&interactive.infoPanelFontSize)->default_value(18), "")

		("scene.fileName", po::value(&scene.fileName)->default_value(""), "")
		("scene.enableTestScenes", po::value(&scene.enableTestScenes)->default_value(true), "")
		("scene.testSceneNumber", po::value(&scene.testSceneNumber)->default_value(1), "")

		("image.width", po::value(&image.width)->default_value(1280), "")
		("image.height", po::value(&image.height)->default_value(800), "")
		("image.fileName", po::value(&image.fileName)->default_value("result.png"), "")
		("image.autoView", po::value(&image.autoView)->default_value(true), "")
		("image.autoWrite", po::value(&image.autoWrite)->default_value(true), "")
		("image.autoWriteInterval", po::value(&image.autoWriteInterval)->default_value(60.0f), "")
		("image.autoWriteCount", po::value(&image.autoWriteCount)->default_value(5), "")
		("image.autoWriteFileName", po::value(&image.autoWriteFileName)->default_value("temp_result_%d.png"), "")

		("window.width", po::value(&window.width)->default_value(1280), "")
		("window.height", po::value(&window.height)->default_value(800), "")
		("window.enableFullscreen", po::value(&window.enableFullscreen)->default_value(false), "")
		("window.enableVsync", po::value(&window.enableVsync)->default_value(false), "")
		("window.hideCursor", po::value(&window.hideCursor)->default_value(false), "")

		("film.restoreFromFile", po::value(&film.restoreFromFile)->default_value(true), "")
		("film.restoreFileName", po::value(&film.restoreFileName)->default_value(""), "")
		("film.autoWrite", po::value(&film.autoWrite)->default_value(true), "")
		("film.autoWriteInterval", po::value(&film.autoWriteInterval)->default_value(60.0f), "")
		("film.autoWriteCount", po::value(&film.autoWriteCount)->default_value(5), "")
		("film.autoWriteFileName", po::value(&film.autoWriteFileName)->default_value("temp_film_%d.bin"), "")

		("camera.enableMovement", po::value(&camera.enableMovement)->default_value(true), "")
		("camera.smoothMovement", po::value(&camera.smoothMovement)->default_value(true), "")
		("camera.freeLook", po::value(&camera.freeLook)->default_value(false), "")
		("camera.moveSpeed", po::value(&camera.moveSpeed)->default_value(10.0f), "")
		("camera.mouseSpeed", po::value(&camera.mouseSpeed)->default_value(40.0f), "")
		("camera.moveDrag", po::value(&camera.moveDrag)->default_value(3.0f), "")
		("camera.mouseDrag", po::value(&camera.mouseDrag)->default_value(6.0f), "")
		("camera.autoStopSpeed", po::value(&camera.autoStopSpeed)->default_value(0.01f), "")
		("camera.slowSpeedModifier", po::value(&camera.slowSpeedModifier)->default_value(0.25f), "")
		("camera.fastSpeedModifier", po::value(&camera.fastSpeedModifier)->default_value(2.5f), "")
		("camera.veryFastSpeedModifier", po::value(&camera.veryFastSpeedModifier)->default_value(5.0f), "");

	std::ifstream iniFile("raycer.ini");
	po::variables_map vm;

	try
	{
		po::store(po::parse_command_line(argc, argv, options), vm);
		po::store(po::parse_config_file(iniFile, options), vm);
		po::notify(vm);
	}
	catch (const po::error& e)
	{
		std::cout << "Command line / settings file parsing failed: " << e.what() << std::endl;
		std::cout << "Try '--help' for list of valid options" << std::endl;

		return false;
	}

	if (vm.count("help"))
	{
		std::cout << options;
		return false;
	}

	return true;
}

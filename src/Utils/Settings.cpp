// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
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

		("general.interactive", po::value(&general.interactive)->default_value(true), "")
		("general.maxThreadCount", po::value(&general.maxThreadCount)->default_value(4), "")
		("general.checkGLErrors", po::value(&general.checkGLErrors)->default_value(true), "")
		
		("network.isClient", po::value(&network.isClient)->default_value(true), "")
		("network.isServer", po::value(&network.isServer)->default_value(true), "")
		("network.localAddress", po::value(&network.localAddress)->default_value(""), "")
		("network.localPort", po::value(&network.localPort)->default_value(45001), "")
		("network.broadcastAddress", po::value(&network.broadcastAddress)->default_value(""), "")
		("network.broadcastPort", po::value(&network.broadcastPort)->default_value(45999), "")

		("scene.fileName", po::value(&scene.fileName)->default_value(""), "")
		("scene.enableTestScenes", po::value(&scene.enableTestScenes)->default_value(true), "")
		("scene.testSceneNumber", po::value(&scene.testSceneNumber)->default_value(1), "")

		("image.width", po::value(&image.width)->default_value(1280), "")
		("image.height", po::value(&image.height)->default_value(800), "")
		("image.fileName", po::value(&image.fileName)->default_value("output.png"), "")
		("image.autoView", po::value(&image.autoView)->default_value(true), "")

		("window.width", po::value(&window.width)->default_value(1280), "")
		("window.height", po::value(&window.height)->default_value(800), "")
		("window.renderScale", po::value(&window.renderScale)->default_value(0.25), "")
		("window.enableFullscreen", po::value(&window.enableFullscreen)->default_value(false), "")
		("window.enableVsync", po::value(&window.enableVsync)->default_value(false), "")
		("window.hideCursor", po::value(&window.hideCursor)->default_value(false), "")
		("window.infoPanelState", po::value(&window.infoPanelState)->default_value(1), "")
		("window.infoPanelFontSize", po::value(&window.infoPanelFontSize)->default_value(14), "")

		("camera.enableMovement", po::value(&camera.enableMovement)->default_value(true), "")
		("camera.smoothMovement", po::value(&camera.smoothMovement)->default_value(true), "")
		("camera.freeLook", po::value(&camera.freeLook)->default_value(false), "")
		("camera.moveSpeed", po::value(&camera.moveSpeed)->default_value(10.0), "")
		("camera.mouseSpeed", po::value(&camera.mouseSpeed)->default_value(40.0), "")
		("camera.moveDrag", po::value(&camera.moveDrag)->default_value(3.0), "")
		("camera.mouseDrag", po::value(&camera.mouseDrag)->default_value(6.0), "")
		("camera.autoStopSpeed", po::value(&camera.autoStopSpeed)->default_value(0.01), "")
		("camera.slowSpeedModifier", po::value(&camera.slowSpeedModifier)->default_value(0.25), "")
		("camera.fastSpeedModifier", po::value(&camera.fastSpeedModifier)->default_value(2.5), "")
		("camera.veryFastSpeedModifier", po::value(&camera.veryFastSpeedModifier)->default_value(5.0), "");

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

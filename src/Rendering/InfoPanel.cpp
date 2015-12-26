// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg/nanovg.h"
#include "nanovg/nanovg_gl.h"

#include "glfw/glfw3.h"

#include "Rendering/InfoPanel.h"
#include "Rendering/Film.h"
#include "Runners/WindowRunner.h"
#include "Scenes/Scene.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"
#include "App.h"

using namespace Raycer;

InfoPanel::~InfoPanel()
{
	if (context != nullptr)
	{
		nvgDeleteGL3(context);
		context = nullptr;
	}
}

void InfoPanel::initialize()
{
	context = nvgCreateGL3(NVG_ANTIALIAS | NVG_STENCIL_STROKES);

	if (context == nullptr)
		throw std::runtime_error("Could not initialize NanoVG");

	nvgCreateFont(context, "mono", "data/fonts/RobotoMono-Regular.ttf");
}

void InfoPanel::render(const Scene& scene, const Film& film)
{
	if (currentState == InfoPanelState::OFF)
		return;

	GLFWwindow* window = App::getWindowRunner().getGlfwWindow();
	int windowWidth, windowHeight, framebufferWidth, framebufferHeight;

	glfwGetWindowSize(window, &windowWidth, &windowHeight);
	glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);

	float pixelRatio = float(framebufferWidth) / float(windowWidth);

	nvgBeginFrame(context, windowWidth, windowHeight, pixelRatio);

	if (currentState == InfoPanelState::FPS)
		renderFps();
	else if (currentState == InfoPanelState::FULL)
		renderFull(scene, film);

	nvgEndFrame(context);
}

void InfoPanel::setState(InfoPanelState state)
{
	currentState = state;
}

void InfoPanel::selectNextState()
{
	if (currentState == InfoPanelState::OFF)
		currentState = InfoPanelState::FPS;
	else if (currentState == InfoPanelState::FPS)
		currentState = InfoPanelState::FULL;
	else if (currentState == InfoPanelState::FULL)
		currentState = InfoPanelState::OFF;
}

void InfoPanel::renderFps()
{
	Settings& settings = App::getSettings();
	const FpsCounter& fpsCounter = App::getWindowRunner().getFpsCounter();

	std::string fpsString = tfm::format("%.1f", fpsCounter.getFps());

	nvgBeginPath(context);

	nvgFontSize(context, float(settings.interactive.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = fpsString.length() * charWidth;
	float panelHeight = bounds[3] - bounds[1];
	float currentX = charWidth / 2.0f;
	float currentY = -bounds[1];

	float rounding = 5.0f;
	float strokeWidth = 2.0f;

	nvgFillColor(context, nvgRGBA(0, 0, 0, 150));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth, panelHeight + rounding, rounding);
	nvgFill(context);

	nvgBeginPath(context);

	nvgStrokeWidth(context, strokeWidth);
	nvgStrokeColor(context, nvgRGBA(0, 0, 0, 180));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth + strokeWidth / 2.0f, panelHeight + rounding + strokeWidth / 2.0f, rounding + 1.0f);
	nvgStroke(context);

	nvgFillColor(context, nvgRGBA(255, 255, 255, 255));
	nvgText(context, currentX, currentY, fpsString.c_str(), nullptr);
}

void InfoPanel::renderFull(const Scene& scene, const Film& film)
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();
	const FpsCounter& fpsCounter = windowRunner.getFpsCounter();

	nvgBeginPath(context);

	nvgFontSize(context, float(settings.interactive.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = 33 * charWidth;
	float panelHeight = 14 * lineSpacing + lineSpacing / 2.0f;
	float currentX = charWidth / 2.0f + 4.0f;
	float currentY = -bounds[1] + 4.0f;

	float rounding = 5.0f;
	float strokeWidth = 2.0f;

	nvgFillColor(context, nvgRGBA(0, 0, 0, 150));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth, panelHeight + rounding, rounding);
	nvgFill(context);

	nvgBeginPath(context);

	nvgStrokeWidth(context, strokeWidth);
	nvgStrokeColor(context, nvgRGBA(0, 0, 0, 180));
	nvgRoundedRect(context, -rounding, -rounding, panelWidth + rounding + charWidth + strokeWidth / 2.0f, panelHeight + rounding + strokeWidth / 2.0f, rounding + 1.0f);
	nvgStroke(context);

	nvgFillColor(context, nvgRGBA(255, 255, 255, 255));

	nvgText(context, currentX, currentY, tfm::format("FPS: %.1f", fpsCounter.getFps()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Frametime: %.1f ms", fpsCounter.getFrameTime()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Position: (%.2f, %.2f, %.2f)", scene.camera.position.x, scene.camera.position.y, scene.camera.position.z).c_str(), nullptr);
	currentY += lineSpacing;

	Vector3 direction = scene.camera.orientation.getDirection();
	nvgText(context, currentX, currentY, tfm::format("Direction: (%.2f, %.2f, %.2f)", direction.x, direction.y, direction.z).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Rotation: (%.2f, %.2f, %.2f)", scene.camera.orientation.pitch, scene.camera.orientation.yaw, scene.camera.orientation.roll).c_str(), nullptr);
	currentY += lineSpacing;

	int64_t scaledMouseX = windowRunner.getMouseInfo().scaledX;
	int64_t scaledMouseY = windowRunner.getMouseInfo().scaledY;
	int64_t scaledMouseIndex = scaledMouseY * film.getWidth() + scaledMouseX;

	nvgText(context, currentX, currentY, tfm::format("Mouse: (%d, %d, %d)", scaledMouseX, scaledMouseY, scaledMouseIndex).c_str(), nullptr);
	currentY += lineSpacing;

	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;
	GLFWwindow* window = windowRunner.getGlfwWindow();

	glfwGetWindowSize(window, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(window, &tempFramebufferWidth, &tempFramebufferHeight);

	nvgText(context, currentX, currentY, tfm::format("Window: %dx%d (%dx%d)", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Film: %dx%d (%.2fx)", film.getWidth(), film.getWidth(), settings.interactive.renderScale).c_str(), nullptr);
	currentY += lineSpacing;

	double pixelsPerSecond = double(film.getWidth() * film.getWidth()) * fpsCounter.getFps();

	nvgText(context, currentX, currentY, tfm::format("Pixels/s: %s", StringUtils::humanizeNumber(pixelsPerSecond)).c_str(), nullptr);
	currentY += lineSpacing;

	double pixelSamplesPerSecond = double(film.getWidth() * film.getWidth()
		* scene.general.pixelSampleCount
		* scene.general.multiSampleCountSqrt
		* scene.general.multiSampleCountSqrt
		* scene.general.timeSampleCount
		* scene.general.cameraSampleCountSqrt
		* scene.general.cameraSampleCountSqrt) * fpsCounter.getFps();

	nvgText(context, currentX, currentY, tfm::format("Pixel samples/s: %s", StringUtils::humanizeNumber(pixelSamplesPerSecond)).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Samples/pixel: %d", film.getSamplesPerPixelCount()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Moving: %s", scene.camera.isMoving()).c_str(), nullptr);
	currentY += lineSpacing;

	std::string tracerName = "unknown";

	switch (scene.general.tracerType)
	{
		case TracerType::RAY: tracerName = "ray"; break;
		case TracerType::PATH: tracerName = "path"; break;
		case TracerType::PREVIEW: tracerName = "preview"; break;
		default: break;
	}

	nvgText(context, currentX, currentY, tfm::format("Tracer: %s", tracerName).c_str(), nullptr);
	currentY += lineSpacing;

	std::string tonemapperName = "unknown";

	switch (scene.tonemapper.type)
	{
		case TonemapperType::PASSTHROUGH: tonemapperName = "passthrough"; break;
		case TonemapperType::LINEAR: tonemapperName = "linear"; break;
		case TonemapperType::SIMPLE: tonemapperName = "simple"; break;
		case TonemapperType::REINHARD: tonemapperName = "reinhard"; break;
		default: break;
	}

	nvgText(context, currentX, currentY, tfm::format("Tonemapper: %s", tonemapperName).c_str(), nullptr);
	currentY += lineSpacing;
}

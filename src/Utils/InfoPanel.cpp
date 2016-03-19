// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#define NANOVG_GL3_IMPLEMENTATION
#include "nanovg/nanovg.h"
#include "nanovg/nanovg_gl.h"

#include "Core/App.h"
#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Runners/WindowRunner.h"
#include "Utils/InfoPanel.h"
#include "Utils/Settings.h"
#include "Utils/StringUtils.h"

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

void InfoPanel::render(const Renderer& renderer, const RenderJob& job)
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
		renderFull(renderer, job);

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

	nvgFontSize(context, float(settings.window.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = fpsString.length() * charWidth;
	float panelHeight = bounds[3] - bounds[1];
	float currentX = charWidth / 2.0f + 2.0f;
	float currentY = -bounds[1] + 2.0f;

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

void InfoPanel::renderFull(const Renderer& renderer, const RenderJob& job)
{
	Settings& settings = App::getSettings();
	WindowRunner& windowRunner = App::getWindowRunner();
	const FpsCounter& fpsCounter = windowRunner.getFpsCounter();
	const Scene& scene = *job.scene;
	const Film& film = *job.film;

	nvgBeginPath(context);

	nvgFontSize(context, float(settings.window.infoPanelFontSize));
	nvgFontFace(context, "mono");

	float lineSpacing;
	nvgTextMetrics(context, nullptr, nullptr, &lineSpacing);
	
	float bounds[4];
	nvgTextBounds(context, 0.0f, 0.0f, "1234567890.", nullptr, bounds);
	float charWidth = (bounds[2] - bounds[0]) / 11.0f;
	float panelWidth = 34 * charWidth;
	float panelHeight = 16 * lineSpacing + lineSpacing / 2.0f;
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

	int64_t filmMouseX = std::max(int64_t(0), std::min(windowRunner.getMouseInfo().filmX, int64_t(film.getWidth() - 1)));
	int64_t filmMouseY = std::max(int64_t(0), std::min(windowRunner.getMouseInfo().filmY, int64_t(film.getHeight() - 1)));;
	int64_t filmMouseIndex = filmMouseY * film.getWidth() + filmMouseX;

	nvgText(context, currentX, currentY, tfm::format("Mouse: (%d, %d, %d)", filmMouseX, filmMouseY, filmMouseIndex).c_str(), nullptr);
	currentY += lineSpacing;

	Color linearColor = film.getLinearColor(filmMouseX, filmMouseY);
	Color outputColor = film.getOutputColor(filmMouseX, filmMouseY);

	nvgText(context, currentX, currentY, tfm::format("Pixel lin: (%.2f, %.2f, %.2f)", linearColor.r, linearColor.g, linearColor.b).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Pixel out: (%.2f, %.2f, %.2f)", outputColor.r, outputColor.g, outputColor.b).c_str(), nullptr);
	currentY += lineSpacing;

	int tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight;
	GLFWwindow* window = windowRunner.getGlfwWindow();

	glfwGetWindowSize(window, &tempWindowWidth, &tempWindowHeight);
	glfwGetFramebufferSize(window, &tempFramebufferWidth, &tempFramebufferHeight);

	nvgText(context, currentX, currentY, tfm::format("Window: %dx%d (%dx%d)", tempWindowWidth, tempWindowHeight, tempFramebufferWidth, tempFramebufferHeight).c_str(), nullptr);
	currentY += lineSpacing;

	float totalPixels = float(film.getWidth() * film.getWidth());

	nvgText(context, currentX, currentY, tfm::format("Film: %dx%d (%.2fx) (%s)", film.getWidth(), film.getWidth(), settings.window.renderScale, StringUtils::humanizeNumber(totalPixels)).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Pixel samples: %d", film.pixelSamples).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Moving: %s", scene.camera.isMoving()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Renderer: %s", renderer.getName()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Integrator: %s", scene.integrator.getName()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Camera: %s", scene.camera.getName()).c_str(), nullptr);
	currentY += lineSpacing;

	nvgText(context, currentX, currentY, tfm::format("Tonemapper: %s", scene.tonemapper.getName()).c_str(), nullptr);
	currentY += lineSpacing;
}

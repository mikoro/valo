// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "tinyformat/tinyformat.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Utils/Settings.h"
#include "Utils/Settings.h"
#include "App.h"

using namespace Valo;

void Renderer::initialize(const Settings& settings)
{
	type = static_cast<RendererType>(settings.renderer.type);
	cpuRenderer.maxThreadCount = settings.general.maxCpuThreadCount;
	imageAutoWrite = settings.image.autoWrite;
	imageAutoWriteInterval = settings.image.autoWriteInterval;
	imageAutoWriteFileName = settings.image.autoWriteFileName;
	filmAutoWrite = settings.film.autoWrite;
	filmAutoWriteInterval = settings.film.autoWriteInterval;
	filmAutoWriteFileName = settings.film.autoWriteFileName;

	cpuRenderer.initialize();
	cudaRenderer.initialize();
}

void Renderer::resize(uint32_t width, uint32_t height)
{
	cpuRenderer.resize(width, height);
	cudaRenderer.resize(width, height);
}

void Renderer::render(RenderJob& job)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;
	Settings& settings = App::getSettings();

	imageAutoWriteTimer.restart();
	filmAutoWriteTimer.restart();

	for (uint32_t i = 0; i < settings.renderer.imageSamples && !job.interrupted; ++i)
	{
		switch (type)
		{
			case RendererType::CPU: cpuRenderer.render(job, filtering); break;
			case RendererType::CUDA: cudaRenderer.render(job, filtering); break;
			default: break;
		}

		film.pixelSamples += settings.renderer.pixelSamples;

		if (imageAutoWrite && imageAutoWriteTimer.getElapsedSeconds() > imageAutoWriteInterval)
		{
			film.normalize(type);
			film.tonemap(scene.tonemapper, type);
			film.getTonemappedImage().download();
			film.getTonemappedImage().save(imageAutoWriteFileName, false);

			imageAutoWriteTimer.restart();
		}

		if (filmAutoWrite && filmAutoWriteTimer.getElapsedSeconds() > filmAutoWriteInterval)
		{
			film.getCumulativeImage().download();
			film.save(filmAutoWriteFileName, false);

			filmAutoWriteTimer.restart();
		}
	}
}

std::string Renderer::getName() const
{
	switch (type)
	{
		case RendererType::CPU: return "cpu";
		case RendererType::CUDA: return "cuda";
		default: return "unknown";
	}
}

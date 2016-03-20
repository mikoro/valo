// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Utils/Settings.h"

using namespace Raycer;

void Renderer::initialize(const Settings& settings)
{
	type = static_cast<RendererType>(settings.renderer.type);
	cpuRenderer.maxThreadCount = settings.renderer.maxCpuThreadCount;
	imageAutoWrite = settings.renderer.imageAutoWrite;
	imageAutoWriteInterval = settings.renderer.imageAutoWriteInterval;
	imageAutoWriteMaxNumber = settings.renderer.imageAutoWriteMaxNumber;
	imageAutoWriteFileName = settings.renderer.imageAutoWriteFileName;

	cpuRenderer.initialize();
	cudaRenderer.initialize();
}

void Renderer::render(RenderJob& job)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	imageAutoWriteTimer.restart();

	for (uint32_t i = 0; i < scene.renderer.pixelSamples && !job.interrupted; ++i)
	{
		switch (type)
		{
			case RendererType::CPU: cpuRenderer.render(job); break;
			case RendererType::CUDA: cudaRenderer.render(job); break;
			default: break;
		}

		++film.pixelSamples;

		if (imageAutoWrite && imageAutoWriteTimer.getElapsedSeconds() > imageAutoWriteInterval)
		{
			film.generateImage(scene.tonemapper);
			film.getImage().save(tfm::format(imageAutoWriteFileName.c_str(), imageAutoWriteNumber), false);

			if (++imageAutoWriteNumber > imageAutoWriteMaxNumber)
				imageAutoWriteNumber = 1;

			imageAutoWriteTimer.restart();
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

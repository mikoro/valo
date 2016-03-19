// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/Renderer.h"
#include "Core/App.h"
#include "Utils/Log.h"

using namespace Raycer;

Renderer Renderer::load(const std::string& fileName)
{
	App::getLog().logInfo("Loading renderer from %s", fileName);

	std::ifstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the renderer file for loading");

	Renderer renderer;

	cereal::XMLInputArchive archive(file);
	archive(renderer);

	file.close();

	return renderer;
}

void Renderer::save(const std::string& fileName) const
{
	App::getLog().logInfo("Saving renderer to %s", fileName);

	std::ofstream file(fileName, std::ios::binary);

	if (!file.good())
		throw std::runtime_error("Could not open the renderer file for saving");

	// force scope
	{
		cereal::XMLOutputArchive archive(file);
		archive(cereal::make_nvp("renderer", *this));
	}

	file.close();
}

void Renderer::initialize()
{
	cpuRenderer.initialize();
	cudaRenderer.initialize();
}

void Renderer::render(RenderJob& job)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	imageAutoWriteTimer.restart();

	for (uint64_t i = 0; i < pixelSamples && !job.interrupted; ++i)
	{
		switch (type)
		{
			case RendererType::CPU: cpuRenderer.render(job); break;
			case RendererType::CUDA: cudaRenderer.render(job); break;
			default: break;
		}

		++film.pixelSamples;

		if (enableImageAutoWrite && imageAutoWriteTimer.getElapsedSeconds() > imageAutoWriteInterval)
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

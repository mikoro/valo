// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

#include "Renderers/CpuRenderer.h"
#include "Renderers/CudaRenderer.h"
#include "Utils/Timer.h"

namespace Valo
{
	class Scene;
	class Film;
	class Settings;

	enum class RendererType { CPU, CUDA };

	struct RenderJob
	{
		RenderJob() : interrupted(false), totalSampleCount(0) {}

		Scene* scene = nullptr;
		Film* film = nullptr;

		std::atomic<bool> interrupted;
		std::atomic<uint32_t> totalSampleCount;
	};

	class Renderer
	{
	public:

		void initialize(const Settings& settings);
		void resize(uint32_t width, uint32_t height);
		void render(RenderJob& job);

		std::string getName() const;

		RendererType type = RendererType::CPU;

		CpuRenderer cpuRenderer;
		CudaRenderer cudaRenderer;

		bool imageAutoWrite = false;
		float imageAutoWriteInterval = 60.0f;
		std::string imageAutoWriteFileName = "temp_image.png";

		bool filmAutoWrite = false;
		float filmAutoWriteInterval = 60.0f;
		std::string filmAutoWriteFileName = "temp_film.bin";

		bool filtering = true;

	private:

		Timer imageAutoWriteTimer;
		Timer filmAutoWriteTimer;
	};
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

#include "Renderers/CpuRenderer.h"
#include "Renderers/CudaRenderer.h"
#include "Utils/Timer.h"

namespace Raycer
{
	class Scene;
	class Film;
	class Settings;

	enum class RendererType { CPU, CUDA };

	struct RenderJob
	{
		RenderJob() : interrupted(false), sampleCount(0) {}

		Scene* scene = nullptr;
		Film* film = nullptr;

		std::atomic<bool> interrupted;
		std::atomic<uint32_t> sampleCount;
	};

	class Renderer
	{
	public:

		void initialize(const Settings& settings);
		void render(RenderJob& job);

		std::string getName() const;

		RendererType type = RendererType::CPU;

		CpuRenderer cpuRenderer;
		CudaRenderer cudaRenderer;

		bool imageAutoWrite = false;
		float imageAutoWriteInterval = 60.0f;
		uint32_t imageAutoWriteMaxNumber = 10;
		std::string imageAutoWriteFileName = "temp_image_%d.png";

		bool filtering = true;

	private:

		Timer imageAutoWriteTimer;
		uint32_t imageAutoWriteNumber = 1;
	};
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

#include "cereal/cereal.hpp"

#include "Renderers/CpuRenderer.h"
#include "Renderers/CudaRenderer.h"
#include "Utils/Timer.h"

namespace Raycer
{
	class Scene;
	class Film;

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

		static Renderer load(const std::string& fileName);
		void save(const std::string& fileName) const;

		void initialize();
		void render(RenderJob& job);

		std::string getName() const;

		RendererType type = RendererType::CPU;
		uint32_t pixelSamples = 1;

		CpuRenderer cpuRenderer;
		CudaRenderer cudaRenderer;

		bool enableImageAutoWrite = false;
		float imageAutoWriteInterval = 60.0f;
		uint32_t imageAutoWriteMaxNumber = 10;
		std::string imageAutoWriteFileName = "temp_image_%d.png";

	private:

		Timer imageAutoWriteTimer;
		uint32_t imageAutoWriteNumber = 1;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(type),
				CEREAL_NVP(pixelSamples),
				CEREAL_NVP(cpuRenderer),
				//CEREAL_NVP(cudaRenderer),
				CEREAL_NVP(enableImageAutoWrite),
				CEREAL_NVP(imageAutoWriteInterval),
				CEREAL_NVP(imageAutoWriteMaxNumber),
				CEREAL_NVP(imageAutoWriteFileName));
		}
	};
}

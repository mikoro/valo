// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <cstdint>

#include "Core/Common.h"
#include "Core/Image.h"
#include "Math/Color.h"
#include "Utils/Allocator.h"

namespace Raycer
{
	class Tonemapper;

	class Film : public Allocator
	{
	public:

		~Film();

		void clear();
		void resize(uint32_t width, uint32_t height);
		CUDA_CALLABLE void addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight);
		CUDA_CALLABLE void addSample(uint32_t index, const Color& color, float filterWeight);

		Color getLinearColor(uint32_t x, uint32_t y) const;
		Color getOutputColor(uint32_t x, uint32_t y) const;
		
		void generateImage(Tonemapper& tonemapper);
		const Image& getImage() const;

		CUDA_CALLABLE uint32_t getWidth() const;
		CUDA_CALLABLE uint32_t getHeight() const;
		CUDA_CALLABLE uint32_t getLength() const;

		bool isCleared() const;

		std::atomic<uint32_t> pixelSamples;

	private:

		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t length = 0;
		
		Color* pixels = nullptr;

		Image linearImage;
		Image outputImage;

		bool cleared = false;
	};
}

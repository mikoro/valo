// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>

#include "Core/Image.h"
#include "Math/Color.h"

namespace Raycer
{
	class Tonemapper;

	class Film
	{
	public:

		~Film();

		void clear();
		void resize(uint64_t width, uint64_t height);
		void addSample(uint64_t x, uint64_t y, const Color& color, float filterWeight);
		void addSample(uint64_t index, const Color& color, float filterWeight);

		Color getLinearColor(uint64_t x, uint64_t y) const;
		Color getOutputColor(uint64_t x, uint64_t y) const;
		
		void generateImage(Tonemapper& tonemapper);
		const Image& getImage() const;

		uint64_t getWidth() const;
		uint64_t getHeight() const;

		std::atomic<uint64_t> pixelSamples;

	private:

		uint64_t width = 0;
		uint64_t height = 0;
		uint64_t length = 0;
		
		Color* pixels = nullptr;

		Image linearImage;
		Image outputImage;
	};
}

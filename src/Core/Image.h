// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/Color.h"

/*

Origin (0, 0) is at the bottom left corner.

*/

namespace Raycer
{
	class Filter;

	class Image
	{
	public:

		Image();
		~Image();

		explicit Image(uint32_t length);
		Image(uint32_t width, uint32_t height);
		Image(uint32_t width, uint32_t height, float* rgbaData);
		explicit Image(const std::string& fileName);

		void load(uint32_t width, uint32_t height, float* rgbaData);
		void load(const std::string& fileName);
		void save(const std::string& fileName, bool writeToLog = true) const;
		void resize(uint32_t length);
		void resize(uint32_t width, uint32_t height);
		void setPixel(uint32_t x, uint32_t y, const Color& color);
		void setPixel(uint32_t index, const Color& color);
		void clear();
		void clear(const Color& color);
		void applyGamma(float gamma);
		void applyFastGamma(float gamma);
		void swapComponents();
		void fillWithTestPattern();

		uint32_t getWidth() const;
		uint32_t getHeight() const;
		uint32_t getLength() const;

		Color getPixel(uint32_t x, uint32_t y) const;
		Color getPixel(uint32_t index) const;
		Color getPixelNearest(float u, float v) const;
		Color getPixelBilinear(float u, float v) const;
		Color getPixelBicubic(float u, float v, Filter& filter) const;

		Color* getPixelData();
		const Color* getPixelData() const;

	private:

		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t length = 0;

		Color* pixels = nullptr;
	};
}

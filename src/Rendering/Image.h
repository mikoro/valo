// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Rendering/Color.h"

/*

Origin (0, 0) is at the bottom left corner.

*/

namespace Raycer
{
	template <typename T>
	class ImageType
	{
	public:

		ImageType();
		explicit ImageType(uint64_t length);
		ImageType(uint64_t width, uint64_t height);
		ImageType(uint64_t width, uint64_t height, float* rgbaData);
		explicit ImageType(const std::string& fileName);

		void load(uint64_t width, uint64_t height, float* rgbaData);
		void load(const std::string& fileName);
		void save(const std::string& fileName) const;
		void resize(uint64_t length);
		void resize(uint64_t width, uint64_t height);
		void setPixel(uint64_t x, uint64_t y, const ColorType<T>& color);
		void setPixel(uint64_t index, const ColorType<T>& color);
		void clear();
		void clear(const ColorType<T>& color);
		void applyGamma(T gamma);
		void applyFastGamma(double gamma);
		void swapComponents();
		void flip();
		void fillTestPattern();

		uint64_t getWidth() const;
		uint64_t getHeight() const;
		uint64_t getLength() const;

		ColorType<T> getPixel(uint64_t x, uint64_t y) const;
		ColorType<T> getPixel(uint64_t index) const;
		ColorType<T> getPixelNearest(double u, double v) const;
		ColorType<T> getPixelBilinear(double u, double v) const;

		std::vector<ColorType<T>>& getPixelData();
		const std::vector<ColorType<T>>& getPixelDataConst() const;

		template <typename U>
		void read(const ImageType<U>& other);

	private:

		uint64_t width = 0;
		uint64_t height = 0;
		uint64_t length = 0;

		std::vector<ColorType<T>> pixelData;
	};

	using Image = ImageType<double>;
	using Imagef = ImageType<float>;
}

#include "Rendering/Image.inl"

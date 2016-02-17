// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include <boost/align/aligned_allocator.hpp>

#include "cereal/cereal.hpp"

#include "Rendering/Color.h"
#include "Common.h"

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
		explicit Image(uint64_t length);
		Image(uint64_t width, uint64_t height);
		Image(uint64_t width, uint64_t height, float* rgbaData);
		explicit Image(const std::string& fileName);

		void load(uint64_t width, uint64_t height, float* rgbaData);
		void load(const std::string& fileName);
		void save(const std::string& fileName, bool writeToLog = true) const;
		void resize(uint64_t length);
		void resize(uint64_t width, uint64_t height);
		void setPixel(uint64_t x, uint64_t y, const Color& color);
		void setPixel(uint64_t index, const Color& color);
		void clear();
		void clear(const Color& color);
		void applyGamma(float gamma);
		void applyFastGamma(float gamma);
		void swapComponents();
		void flip();
		void fillWithTestPattern();

		uint64_t getWidth() const;
		uint64_t getHeight() const;
		uint64_t getLength() const;

		Color getPixel(uint64_t x, uint64_t y) const;
		Color getPixel(uint64_t index) const;
		Color getPixelNearest(float u, float v) const;
		Color getPixelBilinear(float u, float v) const;
		Color getPixelBicubic(float u, float v, Filter* filter) const;

		using vector = std::vector<Color, boost::alignment::aligned_allocator<Color, CACHE_LINE_SIZE>>;

		vector& getPixelData();
		const vector& getPixelDataConst() const;

	private:

		uint64_t width = 0;
		uint64_t height = 0;
		uint64_t length = 0;

		vector pixelData;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(width),
				CEREAL_NVP(height),
				CEREAL_NVP(length),
				CEREAL_NVP(pixelData));
		}
	};
}

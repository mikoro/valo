// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

#include "Rendering/Color.h"
#include "Rendering/Image.h"
#include "Tonemappers/Tonemapper.h"

namespace Raycer
{
	class Scene;

	struct FilmPixel
	{
		Color cumulativeColor = Color(0.0, 0.0, 0.0, 0.0);
		double filterWeightSum = 0.0;
	};

	class Film
	{
	public:

		Film();

		void resize(uint64_t width, uint64_t height);
		void resize(uint64_t length);
		void clear();
		void addSample(uint64_t x, uint64_t y, const Color& color, double filterWeight);
		void addSample(uint64_t index, const Color& color, double filterWeight);
		
		void generateTonemappedImage(const Scene& scene);
		void setTonemappedImage(const Image& other);
		const Image& getTonemappedImage() const;

		uint64_t getWidth() const;
		uint64_t getHeight() const;

	private:

		uint64_t width = 0;
		uint64_t height = 0;

		std::vector<FilmPixel> filmPixels;
		Image linearImage;
		Image tonemappedImage;

		std::map<TonemapperType, std::unique_ptr<Tonemapper>> tonemappers;
	};
}

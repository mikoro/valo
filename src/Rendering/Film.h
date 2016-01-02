// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

#include "cereal/cereal.hpp"

#include "Rendering/Color.h"
#include "Rendering/Image.h"
#include "Tonemappers/Tonemapper.h"

namespace Raycer
{
	class Scene;

	struct FilmPixel
	{
		Color cumulativeColor = Color(0.0, 0.0, 0.0, 0.0);
		double cumulativeFilterWeight = 0.0;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(cumulativeColor),
				CEREAL_NVP(cumulativeFilterWeight));
		}
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
		void increasePixelSamples(uint64_t count);
		void load(const std::string& fileName);
		void save(const std::string& fileName, bool writeToLog = true) const;
		
		void generateOutputImage(const Scene& scene);
		const Image& getOutputImage() const;

		uint64_t getWidth() const;
		uint64_t getHeight() const;
		uint64_t getPixelSamples() const;

	private:

		uint64_t width = 0;
		uint64_t height = 0;
		uint64_t pixelSamples = 0;

		std::vector<FilmPixel> filmPixels;

		Image linearImage;
		Image outputImage;

		std::map<TonemapperType, std::unique_ptr<Tonemapper>> tonemappers;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(width),
				CEREAL_NVP(height),
				CEREAL_NVP(pixelSamples),
				CEREAL_NVP(filmPixels));
		}
	};
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Rendering/Film.h"
#include "Tonemappers/PassthroughTonemapper.h"
#include "Tonemappers/LinearTonemapper.h"
#include "Tonemappers/SimpleTonemapper.h"
#include "Tonemappers/ReinhardTonemapper.h"
#include "Scenes/Scene.h"
#include "App.h"
#include "Utils/Log.h"

using namespace Raycer;

Film::Film()
{
	tonemappers[TonemapperType::PASSTHROUGH] = std::make_unique<PassthroughTonemapper>();
	tonemappers[TonemapperType::LINEAR] = std::make_unique<LinearTonemapper>();
	tonemappers[TonemapperType::SIMPLE] = std::make_unique<SimpleTonemapper>();
	tonemappers[TonemapperType::REINHARD] = std::make_unique<ReinhardTonemapper>();
}

void Film::resize(uint64_t width_, uint64_t height_)
{
	width = width_;
	height = height_;

	App::getLog().logInfo("Resizing film to %sx%s", width, height);

	filmPixels.resize(width * height);
	linearImage.resize(width, height);
	outputImage.resize(width, height);

	clear();
}

void Film::resize(uint64_t length)
{
	resize(length, 1);
}

void Film::clear()
{
	std::memset(&filmPixels[0], 0, filmPixels.size() * sizeof(FilmPixel));
	samplesPerPixelCount = 0;
}

void Film::addSample(uint64_t x, uint64_t y, const Color& color, double filterWeight)
{
	addSample(y * width + x, color, filterWeight);
}

void Film::addSample(uint64_t index, const Color& color, double filterWeight)
{
	FilmPixel& filmPixel = filmPixels[index];
	filmPixel.cumulativeColor += color * filterWeight;
	filmPixel.cumulativeFilterWeight += filterWeight;
}

void Film::increaseSamplesPerPixelCount(uint64_t count)
{
	samplesPerPixelCount += count;
}

void Film::load(const std::string& filePath)
{
	(void)filePath;
}

void Film::save(const std::string& filePath) const
{
	(void)filePath;
}

void Film::generateOutputImage(const Scene& scene)
{
	#pragma omp parallel for
	for (int64_t i = 0; i < int64_t(filmPixels.size()); ++i)
		linearImage.setPixel(i, filmPixels[i].cumulativeColor / filmPixels[i].cumulativeFilterWeight);

	Tonemapper* tonemapper = tonemappers[scene.tonemapping.type].get();
	tonemapper->apply(scene, linearImage, outputImage);
}

const Image& Film::getOutputImage() const
{
	return outputImage;
}

uint64_t Film::getWidth() const
{
	return width;
}

uint64_t Film::getHeight() const
{
	return height;
}

uint64_t Film::getSamplesPerPixelCount() const
{
	return samplesPerPixelCount;
}

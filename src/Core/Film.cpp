// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/App.h"
#include "Core/Film.h"
#include "Tonemappers/Tonemapper.h"
#include "Utils/Log.h"

using namespace Raycer;

Film::~Film()
{
	if (pixels != nullptr)
	{
		free(pixels);
		pixels = nullptr;
	}
}

void Film::clear()
{
	memset(pixels, 0, length * sizeof(Color));
	pixelSamples = 0;
}

void Film::resize(uint32_t width_, uint32_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	App::getLog().logInfo("Resizing film to %sx%s", width, height);

	free(pixels);
	pixels = static_cast<Color*>(malloc(length * sizeof(Color)));

	linearImage.resize(width, height);
	outputImage.resize(width, height);

	clear();
}

void Film::addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight)
{
	addSample(y * width + x, color, filterWeight);
}

void Film::addSample(uint32_t index, const Color& color, float filterWeight)
{
	pixels[index].r += color.r;
	pixels[index].g += color.g;
	pixels[index].b += color.b;
	pixels[index].a += filterWeight;
}

Color Film::getLinearColor(uint32_t x, uint32_t y) const
{
	return linearImage.getPixel(x, y);
}

Color Film::getOutputColor(uint32_t x, uint32_t y) const
{
	return outputImage.getPixel(x, y);
}

void Film::generateImage(Tonemapper& tonemapper)
{
	#pragma omp parallel for
	for (int32_t i = 0; i < int32_t(length); ++i)
		linearImage.setPixel(i, pixels[i] / pixels[i].a);

	tonemapper.apply(linearImage, outputImage);
}

const Image& Film::getImage() const
{
	return outputImage;
}

uint32_t Film::getWidth() const
{
	return width;
}

uint32_t Film::getHeight() const
{
	return height;
}

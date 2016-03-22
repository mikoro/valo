// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Common.h"
#include "App.h"
#include "Core/Film.h"
#include "Tonemappers/Tonemapper.h"
#include "Utils/Log.h"

using namespace Raycer;

Film::~Film()
{
	RAYCER_FREE(pixels);
}

void Film::clear()
{
	memset(pixels, 0, length * sizeof(Color));
	pixelSamples = 0;
	cleared = true;
}

void Film::resize(uint32_t width_, uint32_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	App::getLog().logInfo("Resizing film to %sx%s", width, height);

	RAYCER_FREE(pixels);
	pixels = static_cast<Color*>(RAYCER_MALLOC(length * sizeof(Color)));

	if (pixels == nullptr)
		throw std::runtime_error("Could not allocate memory for film");

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
	pixels[index].r += color.r * filterWeight;
	pixels[index].g += color.g * filterWeight;
	pixels[index].b += color.b * filterWeight;
	pixels[index].a += filterWeight;

	cleared = false;
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
	{
		Color color = pixels[i] / pixels[i].a;
		color.clampPositive();
		color.a = 1.0f;

		linearImage.setPixel(i, color);
	}

	tonemapper.apply(linearImage, outputImage);
}

const Image& Film::getImage() const
{
	return outputImage;
}

CUDA_CALLABLE uint32_t Film::getWidth() const
{
	return width;
}

CUDA_CALLABLE uint32_t Film::getHeight() const
{
	return height;
}

CUDA_CALLABLE uint32_t Film::getLength() const
{
	return length;
}

bool Film::isCleared() const
{
	return cleared;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <vector>

#include "tinyformat/tinyformat.h"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Core/Common.h"
#include "Core/Image.h"
#include "App.h"
#include "Utils/Allocator.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Math/MathUtils.h"
#include "Filters/Filter.h"

using namespace Raycer;

Image::Image()
{
}

Image::~Image()
{
	RAYCER_FREE(pixels);
}

Image::Image(uint32_t length_)
{
	resize(length_);
}

Image::Image(uint32_t width_, uint32_t height_)
{
	resize(width_, height_);
}

Image::Image(uint32_t width_, uint32_t height_, float* rgbaData)
{
	load(width_, height_, rgbaData);
}

Image::Image(const std::string& fileName)
{
	load(fileName);
}

void Image::load(uint32_t width_, uint32_t height_, float* rgbaData)
{
	resize(width_, height_);

	for (uint32_t i = 0; i < length; ++i)
	{
		uint32_t dataIndex = i * 4;

		pixels[i].r = rgbaData[dataIndex];
		pixels[i].g = rgbaData[dataIndex + 1];
		pixels[i].b = rgbaData[dataIndex + 2];
		pixels[i].a = rgbaData[dataIndex + 3];
	}
}

void Image::load(const std::string& fileName)
{
	App::getLog().logInfo("Loading image from %s", fileName);

	if (stbi_is_hdr(fileName.c_str()))
	{
		int32_t newWidth, newHeight, components;
		float* loadData = stbi_loadf(fileName.c_str(), &newWidth, &newHeight, &components, 3); // RGB

		if (loadData == nullptr)
			throw std::runtime_error(tfm::format("Could not load HDR image file: %s", stbi_failure_reason()));

		resize(uint32_t(newWidth), uint32_t(newHeight));

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
			{
				uint32_t pixelIndex = y * width + x;
				uint32_t dataIndex = (height - 1 - y) * width * 3 + x * 3; // flip vertically

				pixels[pixelIndex].r = loadData[dataIndex];
				pixels[pixelIndex].g = loadData[dataIndex + 1];
				pixels[pixelIndex].b = loadData[dataIndex + 2];
				pixels[pixelIndex].a = 1.0f;
			}
		}

		stbi_image_free(loadData);
	}
	else
	{
		int32_t newWidth, newHeight, components;
		uint32_t* loadData = reinterpret_cast<uint32_t*>(stbi_load(fileName.c_str(), &newWidth, &newHeight, &components, 4)); // RGBA

		if (loadData == nullptr)
			throw std::runtime_error(tfm::format("Could not load image file: %s", stbi_failure_reason()));

		resize(uint32_t(newWidth), uint32_t(newHeight));

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
				pixels[y * width + x] = Color::fromAbgrValue(loadData[(height - 1 - y) * width + x]); // flip vertically
		}

		stbi_image_free(loadData);
	}
}

void Image::save(const std::string& fileName, bool writeToLog) const
{
	if (writeToLog)
		App::getLog().logInfo("Saving image to %s", fileName);

	int32_t result = 0;

	if (StringUtils::endsWith(fileName, ".png") || StringUtils::endsWith(fileName, ".bmp") || StringUtils::endsWith(fileName, ".tga"))
	{
		std::vector<uint32_t> saveData(length);

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
				saveData[(height - 1 - y) * width + x] = pixels[y * width + x].clamped().getAbgrValue(); // flip vertically
		}

		if (StringUtils::endsWith(fileName, ".png"))
			result = stbi_write_png(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0], int32_t(width * sizeof(uint32_t)));
		else if (StringUtils::endsWith(fileName, ".bmp"))
			result = stbi_write_bmp(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0]);
		else if (StringUtils::endsWith(fileName, ".tga"))
			result = stbi_write_tga(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0]);
	}
	else if (StringUtils::endsWith(fileName, ".hdr"))
	{
		std::vector<float> saveData(length * 3);

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
			{
				uint32_t dataIndex = (height - 1 - y) * width * 3 + x * 3; // flip vertically
				uint32_t pixelIndex = y * width + x;

				saveData[dataIndex] = float(pixels[pixelIndex].r);
				saveData[dataIndex + 1] = float(pixels[pixelIndex].g);
				saveData[dataIndex + 2] = float(pixels[pixelIndex].b);
			}
		}

		result = stbi_write_hdr(fileName.c_str(), int32_t(width), int32_t(height), 3, &saveData[0]);
	}
	else
		throw std::runtime_error("Could not save the image (non-supported format)");

	if (result == 0)
		throw std::runtime_error(tfm::format("Could not save the image: %s", stbi_failure_reason()));
}

void Image::resize(uint32_t length_)
{
	resize(length_, 1);
}

void Image::resize(uint32_t width_, uint32_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	RAYCER_FREE(pixels);
	pixels = static_cast<Color*>(RAYCER_MALLOC(length * sizeof(Color)));

	if (pixels == nullptr)
		throw std::runtime_error("Could not allocate memory for image");

	clear();
}

void Image::setPixel(uint32_t x, uint32_t y, const Color& color)
{
	pixels[y * width + x] = color;
}

void Image::setPixel(uint32_t index, const Color& color)
{
	pixels[index] = color;
}

void Image::clear()
{
	memset(pixels, 0, length * sizeof(Color));
}

void Image::clear(const Color& color)
{
	for (uint32_t i = 0; i < length; ++i)
		pixels[i] = color;
}

void Image::applyGamma(float gamma)
{
	for (uint32_t i = 0; i < length; ++i)
		pixels[i] = Color::pow(pixels[i], gamma).clamped();
}

void Image::applyFastGamma(float gamma)
{
	for (uint32_t i = 0; i < length; ++i)
		pixels[i] = Color::fastPow(pixels[i], gamma).clamped();
}

void Image::swapComponents()
{
	for (uint32_t i = 0; i < length; ++i)
	{
		Color c2 = pixels[i];

		pixels[i].r = c2.a;
		pixels[i].g = c2.b;
		pixels[i].b = c2.g;
		pixels[i].a = c2.r;
	}
}

void Image::fillWithTestPattern()
{
	for (uint32_t y = 0; y < height; ++y)
	{
		for (uint32_t x = 0; x < width; ++x)
		{
			Color color = Color::black();

			if (x % 2 == 0 && y % 2 == 0)
				color = Color::lerp(Color::red(), Color::blue(), float(x) / float(width));

			pixels[y * width + x] = color;
		}
	}
}

CUDA_CALLABLE uint32_t Image::getWidth() const
{
	return width;
}

CUDA_CALLABLE uint32_t Image::getHeight() const
{
	return height;
}

CUDA_CALLABLE uint32_t Image::getLength() const
{
	return length;
}

CUDA_CALLABLE Color Image::getPixel(uint32_t x, uint32_t y) const
{
	assert(x < width && y < height);

	return pixels[y * width + x];
}

CUDA_CALLABLE Color Image::getPixel(uint32_t index) const
{
	assert(index < length);

	return pixels[index];
}

CUDA_CALLABLE Color Image::getPixelNearest(float u, float v) const
{
	uint32_t x = uint32_t(u * float(width - 1) + 0.5f);
	uint32_t y = uint32_t(v * float(height - 1) + 0.5f);

	return getPixel(x, y);
}

CUDA_CALLABLE Color Image::getPixelBilinear(float u, float v) const
{
	float x = u * float(width - 1);
	float y = v * float(height - 1);

	uint32_t ix = uint32_t(x);
	uint32_t iy = uint32_t(y);

	float tx2 = x - float(ix);
	float ty2 = y - float(iy);

	tx2 = MathUtils::smoothstep(tx2);
	ty2 = MathUtils::smoothstep(ty2);

	float tx1 = 1.0f - tx2;
	float ty1 = 1.0f - ty2;

	uint32_t ix1 = ix + 1;
	uint32_t iy1 = iy + 1;

	if (ix1 > width - 1)
		ix1 = width - 1;

	if (iy1 > height - 1)
		iy1 = height - 1;

	Color c11 = getPixel(ix, iy);
	Color c21 = getPixel(ix1, iy);
	Color c12 = getPixel(ix, iy1);
	Color c22 = getPixel(ix1, iy1);

	// bilinear interpolation
	return (tx1 * c11 + tx2 * c21) * ty1 + (tx1 * c12 + tx2 * c22) * ty2;
}

CUDA_CALLABLE Color Image::getPixelBicubic(float u, float v, Filter& filter) const
{
	float x = u * float(width - 1);
	float y = v * float(height - 1);

	int32_t ix = int32_t(x);
	int32_t iy = int32_t(y);

	float fx = x - float(ix);
	float fy = y - float(iy);

	Color cumulativeColor;
	float cumulativeFilterWeight = 0.0f;

	for (int32_t oy = -1; oy <= 2; oy++)
	{
		for (int32_t ox = -1; ox <= 2; ox++)
		{
			int32_t sx = ix + ox;
			int32_t sy = iy + oy;

			if (sx < 0)
				sx = 0;

			if (sx > int32_t(width - 1))
				sx = int32_t(width - 1);

			if (sy < 0)
				sy = 0;

			if (sy > int32_t(height - 1))
				sy = int32_t(height - 1);

			Color color = getPixel(uint32_t(sx), uint32_t(sy));
			float filterWeight = filter.getWeight(Vector2(float(ox) - fx, float(oy) - fy));

			cumulativeColor += color * filterWeight;
			cumulativeFilterWeight += filterWeight;
		}
	}

	return cumulativeColor / cumulativeFilterWeight;
}

Color* Image::getPixelData()
{
	return pixels;
}

const Color* Image::getPixelData() const
{
	return pixels;
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Rendering/Image.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Math/MathUtils.h"
#include "Filters/Filter.h"

using namespace Raycer;

Image::Image()
{
}

Image::Image(uint64_t length_)
{
	resize(length_);
}

Image::Image(uint64_t width_, uint64_t height_)
{
	resize(width_, height_);
}

Image::Image(uint64_t width_, uint64_t height_, float* rgbaData)
{
	load(width_, height_, rgbaData);
}

Image::Image(const std::string& fileName)
{
	load(fileName);
}

void Image::load(uint64_t width_, uint64_t height_, float* rgbaData)
{
	resize(width_, height_);

	for (uint64_t i = 0; i < pixelData.size(); ++i)
	{
		uint64_t dataIndex = i * 4;

		pixelData[i].r = rgbaData[dataIndex];
		pixelData[i].g = rgbaData[dataIndex + 1];
		pixelData[i].b = rgbaData[dataIndex + 2];
		pixelData[i].a = rgbaData[dataIndex + 3];
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

		resize(uint64_t(newWidth), uint64_t(newHeight));

		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
			{
				uint64_t pixelIndex = y * width + x;
				uint64_t dataIndex = (height - 1 - y) * width * 3 + x * 3; // flip vertically

				pixelData[pixelIndex].r = loadData[dataIndex];
				pixelData[pixelIndex].g = loadData[dataIndex + 1];
				pixelData[pixelIndex].b = loadData[dataIndex + 2];
				pixelData[pixelIndex].a = 1.0;
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

		resize(uint64_t(newWidth), uint64_t(newHeight));

		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
				pixelData[y * width + x] = Color::fromAbgrValue(loadData[(height - 1 - y) * width + x]); // flip vertically
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
		std::vector<uint32_t> saveData(pixelData.size());

		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
				saveData[(height - 1 - y) * width + x] = pixelData[y * width + x].clamped().getAbgrValue(); // flip vertically
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
		std::vector<float> saveData(pixelData.size() * 3);

		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
			{
				uint64_t dataIndex = (height - 1 - y) * width * 3 + x * 3; // flip vertically
				uint64_t pixelIndex = y * width + x;

				saveData[dataIndex] = float(pixelData[pixelIndex].r);
				saveData[dataIndex + 1] = float(pixelData[pixelIndex].g);
				saveData[dataIndex + 2] = float(pixelData[pixelIndex].b);
			}
		}

		result = stbi_write_hdr(fileName.c_str(), int32_t(width), int32_t(height), 3, &saveData[0]);
	}
	else
		throw std::runtime_error("Could not save the image (non-supported format)");

	if (result == 0)
		throw std::runtime_error(tfm::format("Could not save the image: %s", stbi_failure_reason()));
}

void Image::resize(uint64_t length_)
{
	resize(length_, 1);
}

void Image::resize(uint64_t width_, uint64_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	pixelData.resize(length);
	clear();
}

void Image::setPixel(uint64_t x, uint64_t y, const Color& color)
{
	pixelData[y * width + x] = color;
}

void Image::setPixel(uint64_t index, const Color& color)
{
	pixelData[index] = color;
}

void Image::clear()
{
	for (Color& c : pixelData)
		c = Color();
}

void Image::clear(const Color& color)
{
	for (Color& c : pixelData)
		c = color;
}

void Image::applyGamma(float gamma)
{
	for (Color& c : pixelData)
		c = Color::pow(c, gamma).clamped();
}

void Image::applyFastGamma(float gamma)
{
	for (Color& c : pixelData)
		c = Color::fastPow(c, gamma).clamped();
}

void Image::swapComponents()
{
	for (Color& c1 : pixelData)
	{
		Color c2 = c1;

		c1.r = c2.a;
		c1.g = c2.b;
		c1.b = c2.g;
		c1.a = c2.r;
	}
}

void Image::flip()
{
	Image tempImage(width, height);

	for (uint64_t y = 0; y < height; ++y)
	{
		for (uint64_t x = 0; x < width; ++x)
			tempImage.pixelData[(height - 1 - y) * width + x] = pixelData[y * width + x];
	}

	*this = tempImage;
}

void Image::fillWithTestPattern()
{
	for (uint64_t y = 0; y < height; ++y)
	{
		for (uint64_t x = 0; x < width; ++x)
		{
			Color color = Color::BLACK;

			if (x % 2 == 0 && y % 2 == 0)
				color = Color::lerp(Color::RED, Color::BLUE, float(x) / float(width));

			pixelData[y * width + x] = color;
		}
	}
}

uint64_t Image::getWidth() const
{
	return width;
}

uint64_t Image::getHeight() const
{
	return height;
}

uint64_t Image::getLength() const
{
	return length;
}

Color Image::getPixel(uint64_t x, uint64_t y) const
{
	assert(x < width && y < height);
	return pixelData[y * width + x];
}

Color Image::getPixel(uint64_t index) const
{
	assert(index < length);
	return pixelData[index];
}

Color Image::getPixelNearest(float u, float v) const
{
	uint64_t x = uint64_t(u * float(width - 1) + 0.5f);
	uint64_t y = uint64_t(v * float(height - 1) + 0.5f);

	return getPixel(x, y);
}

Color Image::getPixelBilinear(float u, float v) const
{
	float x = u * float(width - 1);
	float y = v * float(height - 1);

	uint64_t ix = uint64_t(x);
	uint64_t iy = uint64_t(y);

	float tx2 = x - float(ix);
	float ty2 = y - float(iy);

	tx2 = MathUtils::smoothstep(tx2);
	ty2 = MathUtils::smoothstep(ty2);

	float tx1 = 1.0f - tx2;
	float ty1 = 1.0f - ty2;

	uint64_t ix1 = ix + 1;
	uint64_t iy1 = iy + 1;

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

Color Image::getPixelBicubic(float u, float v, Filter* filter) const
{
	float x = u * float(width - 1);
	float y = v * float(height - 1);

	int64_t ix = int64_t(x);
	int64_t iy = int64_t(y);

	float fx = x - float(ix);
	float fy = y - float(iy);

	Color cumulativeColor;
	float cumulativeFilterWeight = 0.0f;

	for (int64_t oy = -1; oy <= 2; oy++)
	{
		for (int64_t ox = -1; ox <= 2; ox++)
		{
			int64_t sx = ix + ox;
			int64_t sy = iy + oy;

			if (sx < 0)
				sx = 0;

			if (sx > int64_t(width - 1))
				sx = int64_t(width - 1);

			if (sy < 0)
				sy = 0;

			if (sy > int64_t(height - 1))
				sy = int64_t(height - 1);

			Color color = getPixel(uint64_t(sx), uint64_t(sy));
			float filterWeight = filter->getWeight(float(ox) - fx, float(oy) - fy);

			cumulativeColor += color * filterWeight;
			cumulativeFilterWeight += filterWeight;
		}
	}

	return cumulativeColor / cumulativeFilterWeight;
}

Image::vector& Image::getPixelData()
{
	return pixelData;
}

const Image::vector& Image::getPixelDataConst() const
{
	return pixelData;
}

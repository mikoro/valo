// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Math/MathUtils.h"

namespace Raycer
{
	template <typename T>
	ImageType<T>::ImageType()
	{
	}

	template <typename T>
	ImageType<T>::ImageType(uint64_t length_)
	{
		resize(length_);
	}

	template <typename T>
	ImageType<T>::ImageType(uint64_t width_, uint64_t height_)
	{
		resize(width_, height_);
	}

	template <typename T>
	ImageType<T>::ImageType(uint64_t width_, uint64_t height_, float* rgbaData)
	{
		load(width_, height_, rgbaData);
	}

	template <typename T>
	ImageType<T>::ImageType(const std::string& fileName)
	{
		load(fileName);
	}

	template <typename T>
	void ImageType<T>::load(uint64_t width_, uint64_t height_, float* rgbaData)
	{
		resize(width_, height_);

		for (uint64_t i = 0; i < pixelData.size(); ++i)
		{
			uint64_t dataIndex = i * 4;

			pixelData[i].r = T(rgbaData[dataIndex]);
			pixelData[i].g = T(rgbaData[dataIndex + 1]);
			pixelData[i].b = T(rgbaData[dataIndex + 2]);
			pixelData[i].a = T(rgbaData[dataIndex + 3]);
		}
	}

	template <typename T>
	void ImageType<T>::load(const std::string& fileName)
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

					pixelData[pixelIndex].r = T(loadData[dataIndex]);
					pixelData[pixelIndex].g = T(loadData[dataIndex + 1]);
					pixelData[pixelIndex].b = T(loadData[dataIndex + 2]);
					pixelData[pixelIndex].a = T(1.0);
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
					pixelData[y * width + x] = ColorType<T>::fromAbgrValue(loadData[(height - 1 - y) * width + x]); // flip vertically
			}

			stbi_image_free(loadData);
		}
	}

	template <typename T>
	void ImageType<T>::save(const std::string& fileName) const
	{
		App::getLog().logInfo("Saving image to %s", fileName);

		int32_t result = 0;

		if (StringUtils::endsWith(fileName, ".png") || StringUtils::endsWith(fileName, ".bmp") || StringUtils::endsWith(fileName, ".tga"))
		{
			std::vector<uint32_t> saveData(pixelData.size());

			for (uint64_t y = 0; y < height; ++y)
			{
				for (uint64_t x = 0; x < width; ++x)
					saveData[(height - 1 - y) * width + x] = pixelData[y * width + x].getAbgrValue(); // flip vertically
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

	template <typename T>
	void ImageType<T>::resize(uint64_t length_)
	{
		resize(length_, 1);
	}

	template <typename T>
	void ImageType<T>::resize(uint64_t width_, uint64_t height_)
	{
		width = width_;
		height = height_;

		pixelData.resize(width * height);
		clear();
	}

	template <typename T>
	void ImageType<T>::setPixel(uint64_t x, uint64_t y, const ColorType<T>& color)
	{
		pixelData[y * width + x] = color;
	}

	template <typename T>
	void ImageType<T>::setPixel(uint64_t index, const ColorType<T>& color)
	{
		pixelData[index] = color;
	}

	template <typename T>
	void ImageType<T>::clear()
	{
		for (ColorType<T>& c : pixelData)
			c = ColorType<T>();
	}

	template <typename T>
	void ImageType<T>::clear(const ColorType<T>& color)
	{
		for (ColorType<T>& c : pixelData)
			c = color;
	}

	template <typename T>
	void ImageType<T>::applyGamma(T gamma)
	{
		for (ColorType<T>& c : pixelData)
			c = ColorType<T>::pow(c, gamma).clamped();
	}

	template <typename T>
	void ImageType<T>::applyFastGamma(double gamma)
	{
		for (ColorType<T>& c : pixelData)
			c = ColorType<T>::fastPow(c, gamma).clamped();
	}

	template <typename T>
	void ImageType<T>::swapComponents()
	{
		for (ColorType<T>& c1 : pixelData)
		{
			ColorType<T> c2 = c1;

			c1.r = c2.a;
			c1.g = c2.b;
			c1.b = c2.g;
			c1.a = c2.r;
		}
	}

	template <typename T>
	void ImageType<T>::flip()
	{
		ImageType<T> tempImage(width, height);

		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
				tempImage.pixelData[(height - 1 - y) * width + x] = pixelData[y * width + x];
		}

		*this = tempImage;
	}

	template <typename T>
	void ImageType<T>::fillTestPattern()
	{
		for (uint64_t y = 0; y < height; ++y)
		{
			for (uint64_t x = 0; x < width; ++x)
			{
				ColorType<T> color = ColorType<T>::BLACK;

				if (x % 2 == 0 && y % 2 == 0)
					color = ColorType<T>::lerp(ColorType<T>::RED, ColorType<T>::BLUE, T(double(x) / double(width)));

				pixelData[y * width + x] = color;
			}
		}
	}

	template <typename T>
	uint64_t ImageType<T>::getWidth() const
	{
		return width;
	}

	template <typename T>
	uint64_t ImageType<T>::getHeight() const
	{
		return height;
	}

	template <typename T>
	uint64_t ImageType<T>::getLength() const
	{
		return width * height;
	}

	template <typename T>
	ColorType<T> ImageType<T>::getPixel(uint64_t x, uint64_t y) const
	{
		assert(x < width && y < height);
		return pixelData[y * width + x];
	}

	template <typename T>
	ColorType<T> ImageType<T>::getPixel(uint64_t index) const
	{
		assert(index < width * height);
		return pixelData[index];
	}

	template <typename T>
	ColorType<T> ImageType<T>::getPixelNearest(double u, double v) const
	{
		uint64_t x = uint64_t(u * double(width - 1) + 0.5);
		uint64_t y = uint64_t(v * double(height - 1) + 0.5);

		return getPixel(x, y);
	}

	template <typename T>
	ColorType<T> ImageType<T>::getPixelBilinear(double u, double v) const
	{
		double dx = u * double(width - 1) - 0.5;
		double dy = v * double(height - 1) - 0.5;
		uint64_t ix = uint64_t(dx);
		uint64_t iy = uint64_t(dy);
		double tx2 = dx - double(ix);
		double ty2 = dy - double(iy);
		tx2 = MathUtils::smoothstep(tx2);
		ty2 = MathUtils::smoothstep(ty2);
		double tx1 = 1.0 - tx2;
		double ty1 = 1.0 - ty2;

		ColorType<T> c11 = getPixel(ix, iy);
		ColorType<T> c21 = getPixel(ix + 1, iy);
		ColorType<T> c12 = getPixel(ix, iy + 1);
		ColorType<T> c22 = getPixel(ix + 1, iy + 1);

		// bilinear interpolation
		return (T(tx1) * c11 + T(tx2) * c21) * T(ty1) + (T(tx1) * c12 + T(tx2) * c22) * T(ty2);
	}

	template <typename T>
	std::vector<ColorType<T>>& ImageType<T>::getPixelData()
	{
		return pixelData;
	}

	template <typename T>
	const std::vector<ColorType<T>>& ImageType<T>::getPixelDataConst() const
	{
		return pixelData;
	}
}

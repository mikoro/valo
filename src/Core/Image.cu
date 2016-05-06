// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef USE_CUDA
#include <device_launch_parameters.h>
#endif

#include "tinyformat/tinyformat.h"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Core/Common.h"
#include "Core/Image.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/StringUtils.h"
#include "Utils/CudaUtils.h"
#include "Math/MathUtils.h"
#include "Filters/Filter.h"
#include "Renderers/Renderer.h"
#include "Utils/SysUtils.h"

using namespace Raycer;

Image::Image()
{
}

Image::~Image()
{
	if (data != nullptr)
	{
		free(data);
		data = nullptr;
	}

#ifdef USE_CUDA

	if (textureObject != 0)
	{
		CudaUtils::checkError(cudaDestroyTextureObject(textureObject), "Could not destroy texture object");
		textureObject = 0;
	}

	if (surfaceObject != 0)
	{
		CudaUtils::checkError(cudaDestroySurfaceObject(surfaceObject), "Could not destroy surface object");
		surfaceObject = 0;
	}

	if (cudaData != nullptr)
	{
		CudaUtils::checkError(cudaFreeArray(cudaData), "Could not free array");
		cudaData = nullptr;
	}

#endif
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

Image::Image(const Image& other)
{
	resize(other.width, other.height);

	for (uint32_t i = 0; i < length; ++i)
		data[i] = other.data[i];
}

Image& Image::operator=(const Image& other)
{
	resize(other.width, other.height);

	for (uint32_t i = 0; i < length; ++i)
		data[i] = other.data[i];

	return *this;
}

void Image::load(uint32_t width_, uint32_t height_, float* rgbaData)
{
	resize(width_, height_);

	for (uint32_t i = 0; i < length; ++i)
	{
		uint32_t dataIndex = i * 4;

		data[i].r = rgbaData[dataIndex];
		data[i].g = rgbaData[dataIndex + 1];
		data[i].b = rgbaData[dataIndex + 2];
		data[i].a = rgbaData[dataIndex + 3];
	}
}

void Image::load(const std::string& fileName)
{
	App::getLog().logInfo("Loading image from %s", fileName);

	if (StringUtils::endsWith(fileName, ".jpg") || StringUtils::endsWith(fileName, ".png") || StringUtils::endsWith(fileName, ".bmp") || StringUtils::endsWith(fileName, ".tga"))
	{
		int32_t newWidth, newHeight, components;
		uint32_t* loadData = reinterpret_cast<uint32_t*>(stbi_load(fileName.c_str(), &newWidth, &newHeight, &components, 4)); // RGBA

		if (loadData == nullptr)
			throw std::runtime_error(tfm::format("Could not load image file: %s", stbi_failure_reason()));

		resize(uint32_t(newWidth), uint32_t(newHeight));

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
				data[y * width + x] = Color::fromAbgrValue(loadData[(height - 1 - y) * width + x]); // flip vertically
		}

		stbi_image_free(loadData);
	}
	else if (StringUtils::endsWith(fileName, ".hdr"))
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

				data[pixelIndex].r = loadData[dataIndex];
				data[pixelIndex].g = loadData[dataIndex + 1];
				data[pixelIndex].b = loadData[dataIndex + 2];
				data[pixelIndex].a = 1.0f;
			}
		}

		stbi_image_free(loadData);
	}
	else if (StringUtils::endsWith(fileName, ".bin"))
	{
		uint64_t fileSize = SysUtils::getFileSize(fileName);

		if (fileSize < 8)
			throw std::runtime_error("Binary image file is not valid (too small)");

		std::ifstream file(fileName, std::ios::in | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("Could not open the binary file for reading");

		uint32_t newWidth, newHeight;
		file.read(reinterpret_cast<char*>(&newWidth), 4);
		file.read(reinterpret_cast<char*>(&newHeight), 4);

		if (fileSize != (width * height * sizeof(Color) + 8))
			throw std::runtime_error("Binary image file is not valid (wrong size)");

		resize(newWidth, newHeight);
		file.read(reinterpret_cast<char*>(data), fileSize - 8);
		file.close();
	}
	else
		throw std::runtime_error("Could not load the image (non-supported format)");
}

void Image::save(const std::string& fileName, bool writeToLog) const
{
	if (writeToLog)
		App::getLog().logInfo("Saving image to %s", fileName);

	if (StringUtils::endsWith(fileName, ".png") || StringUtils::endsWith(fileName, ".bmp") || StringUtils::endsWith(fileName, ".tga"))
	{
		std::vector<uint32_t> saveData(length);

		for (uint32_t y = 0; y < height; ++y)
		{
			for (uint32_t x = 0; x < width; ++x)
				saveData[(height - 1 - y) * width + x] = data[y * width + x].clamped().getAbgrValue(); // flip vertically
		}

		int32_t result = 0;

		if (StringUtils::endsWith(fileName, ".png"))
			result = stbi_write_png(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0], int32_t(width * sizeof(uint32_t)));
		else if (StringUtils::endsWith(fileName, ".bmp"))
			result = stbi_write_bmp(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0]);
		else if (StringUtils::endsWith(fileName, ".tga"))
			result = stbi_write_tga(fileName.c_str(), int32_t(width), int32_t(height), 4, &saveData[0]);

		if (result == 0)
			throw std::runtime_error(tfm::format("Could not save the image: %s", stbi_failure_reason()));
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

				saveData[dataIndex] = float(data[pixelIndex].r);
				saveData[dataIndex + 1] = float(data[pixelIndex].g);
				saveData[dataIndex + 2] = float(data[pixelIndex].b);
			}
		}

		int32_t result = stbi_write_hdr(fileName.c_str(), int32_t(width), int32_t(height), 3, &saveData[0]);

		if (result == 0)
			throw std::runtime_error(tfm::format("Could not save the image: %s", stbi_failure_reason()));
	}
	else if (StringUtils::endsWith(fileName, ".bin"))
	{
		std::ofstream file(fileName, std::ios::out | std::ios::binary);

		if (!file.is_open())
			throw std::runtime_error("Could not open the binary file for writing");

		file.write(reinterpret_cast<const char*>(&width), 4);
		file.write(reinterpret_cast<const char*>(&height), 4);
		file.write(reinterpret_cast<const char*>(data), sizeof(Color) * length);

		file.close();
	}
	else
		throw std::runtime_error("Could not save the image (non-supported format)");
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

	if (data != nullptr)
	{
		free(data);
		data = nullptr;
	}

	data = static_cast<Color*>(malloc(length * sizeof(Color)));

	if (data == nullptr)
		throw std::runtime_error("Could not allocate memory for image");

#ifdef USE_CUDA

	if (textureObject != 0)
	{
		CudaUtils::checkError(cudaDestroyTextureObject(textureObject), "Could not destroy texture object");
		textureObject = 0;
	}

	if (surfaceObject != 0)
	{
		CudaUtils::checkError(cudaDestroySurfaceObject(surfaceObject), "Could not surface texture object");
		surfaceObject = 0;
	}

	if (cudaData != nullptr)
	{
		CudaUtils::checkError(cudaFreeArray(cudaData), "Could not free array");
		cudaData = nullptr;
	}

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CudaUtils::checkError(cudaMallocArray(&cudaData, &channelDesc, width, height, cudaArraySurfaceLoadStore), "Could not allocate memory");

	if (cudaData == nullptr)
		throw std::runtime_error("Could not allocate cuda memory for image");

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = cudaData;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	CudaUtils::checkError(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, nullptr), "Could not create texture object");
	CudaUtils::checkError(cudaCreateSurfaceObject(&surfaceObject, &resDesc), "Could not create surface object");

#endif
}

#ifdef USE_CUDA

__global__ void clearKernel(cudaSurfaceObject_t surfaceObject, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	surf2Dwrite(color, surfaceObject, x * sizeof(float4), y);
}

#endif

void Image::clear(RendererType type)
{
	if (type == RendererType::CPU)
		memset(data, 0, length * sizeof(Color));
	else
	{
#ifdef USE_CUDA

		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		clearKernel<<<dimGrid, dimBlock>>>(surfaceObject, width, height);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch clear kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute clear kernel");
#else
		memset(data, 0, length * sizeof(Color));
#endif
	}
}

void Image::clear(const Color& color)
{
	for (uint32_t i = 0; i < length; ++i)
		data[i] = color;
}

void Image::applyGamma(float gamma)
{
	for (uint32_t i = 0; i < length; ++i)
		data[i] = Color::pow(data[i], gamma).clamped();
}

void Image::applyFastGamma(float gamma)
{
	for (uint32_t i = 0; i < length; ++i)
		data[i] = Color::fastPow(data[i], gamma).clamped();
}

void Image::swapComponents()
{
	for (uint32_t i = 0; i < length; ++i)
	{
		Color c2 = data[i];

		data[i].r = c2.a;
		data[i].g = c2.b;
		data[i].b = c2.g;
		data[i].a = c2.r;
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

			data[y * width + x] = color;
		}
	}
}

CUDA_CALLABLE void Image::setPixel(uint32_t x, uint32_t y, const Color& color)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

	surf2Dwrite(make_float4(color.r, color.g, color.b, color.a), surfaceObject, x * sizeof(float4), y);

#else

	assert(x < width && y < height);
	data[y * width + x] = color;

#endif
}

CUDA_CALLABLE void Image::setPixel(uint32_t index, const Color& color)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

	surf1Dwrite(make_float4(color.r, color.g, color.b, color.a), surfaceObject, index * sizeof(float4));

#else

	assert(index < length);
	data[index] = color;

#endif
}

CUDA_CALLABLE Color Image::getPixel(uint32_t x, uint32_t y) const
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

	float4 color;
	surf2Dread(&color, surfaceObject, x * sizeof(float4), y);
	return Color(color.x, color.y, color.z, color.w);

#else

	assert(x < width && y < height);
	return data[y * width + x];

#endif
}

CUDA_CALLABLE Color Image::getPixel(uint32_t index) const
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

	float4 color;
	surf1Dread(&color, surfaceObject, index * sizeof(float4));
	return Color(color.x, color.y, color.z, color.w);

#else

	assert(index < length);
	return data[index];

#endif
}

CUDA_CALLABLE Color Image::getPixelNearest(float u, float v) const
{
	uint32_t x = uint32_t(u * float(width - 1) + 0.5f);
	uint32_t y = uint32_t(v * float(height - 1) + 0.5f);

	return getPixel(x, y);
}

CUDA_CALLABLE Color Image::getPixelBilinear(float u, float v) const
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))

	float4 color = tex2D<float4>(textureObject, u, v);
	return Color(color.x, color.y, color.z, color.w);

#else

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

#endif
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

void Image::upload()
{
#ifdef USE_CUDA
	CudaUtils::checkError(cudaMemcpyToArray(cudaData, 0, 0, data, length * sizeof(Color), cudaMemcpyHostToDevice), "Could not upload image to device");
#endif
}

void Image::download()
{
#ifdef USE_CUDA
	CudaUtils::checkError(cudaMemcpyFromArray(data, cudaData, 0, 0, length * sizeof(Color), cudaMemcpyDeviceToHost), "Could not download image from device");
#endif
}

Color* Image::getData()
{
	return data;
}

const Color* Image::getData() const
{
	return data;
}

#ifdef USE_CUDA

CUDA_CALLABLE cudaSurfaceObject_t Image::getSurfaceObject() const
{
	return surfaceObject;
}

#endif

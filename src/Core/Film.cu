// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <GL/gl3w.h>

#ifdef USE_CUDA
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#endif

#include "Core/Common.h"
#include "App.h"
#include "Core/Film.h"
#include "Tonemappers/Tonemapper.h"
#include "Utils/Log.h"
#include "Utils/CudaUtils.h"
#include "Utils/GLUtils.h"

using namespace Raycer;

Film::Film(bool windowed_) : windowed(windowed_)
{
}

void Film::initialize()
{
	if (windowed)
	{
		glGenTextures(1, &textureId);

		GLUtils::checkError("Could not create OpenGL texture");

		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		GLUtils::checkError("Could not set OpenGL texture parameters");
	}
}

void Film::shutdown()
{
	if (windowed)
	{
#ifdef USE_CUDA

		if (textureResource != nullptr)
		{
			CudaUtils::checkError(cudaGraphicsUnregisterResource(textureResource), "Could not unregister OpenGL texture");
			textureResource = nullptr;
		}

#endif

		glDeleteTextures(1, &textureId);

		GLUtils::checkError("Could not delete OpenGL texture");
	}
}

void Film::resize(uint32_t width_, uint32_t height_)
{
	width = width_;
	height = height_;
	length = width * height;

	App::getLog().logInfo("Resizing film to %sx%s", width, height);

	cumulativeImage.resize(width, height);
	normalizedImage.resize(width, height);
	tonemappedImage.resize(width, height);

	if (windowed)
	{
#ifdef USE_CUDA

		if (textureResource != nullptr)
		{
			CudaUtils::checkError(cudaGraphicsUnregisterResource(textureResource), "Could not unregister OpenGL texture");
			textureResource = nullptr;
		}

#endif

		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GLsizei(width), GLsizei(height), 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);

		GLUtils::checkError("Could not reserve OpenGL texture memory");

#ifdef USE_CUDA
		CudaUtils::checkError(cudaGraphicsGLRegisterImage(&textureResource, textureId, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore), "Could not register OpenGL texture");
#endif
	}
}

void Film::clear(RendererType type)
{
	cumulativeImage.clear(type);
	pixelSamples = 0;
	cleared = true;
}

bool Film::hasBeenCleared() const
{
	return cleared;
}

void Film::resetCleared()
{
	cleared = false;
}

CUDA_CALLABLE void Film::addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight)
{
	Color temp = cumulativeImage.getPixel(x, y);

	temp.r += color.r * filterWeight;
	temp.g += color.g * filterWeight;
	temp.b += color.b * filterWeight;
	temp.a += filterWeight;

	cumulativeImage.setPixel(x, y, temp);
}

CUDA_CALLABLE void Film::addSample(uint32_t index, const Color& color, float filterWeight)
{
	Color temp = cumulativeImage.getPixel(index);

	temp.r += color.r * filterWeight;
	temp.g += color.g * filterWeight;
	temp.b += color.b * filterWeight;
	temp.a += filterWeight;

	cumulativeImage.setPixel(index, temp);
}

#ifdef USE_CUDA

__global__ void normalizeKernel(cudaSurfaceObject_t cumulative, cudaSurfaceObject_t normalized, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color;
	surf2Dread(&color, cumulative, x * sizeof(float4), y);

	color.x /= color.w;
	color.y /= color.w;
	color.z /= color.w;
	color.w = 1.0f;

	surf2Dwrite(color, normalized, x * sizeof(float4), y);
}

#endif

void Film::normalize(RendererType type)
{
	if (type == RendererType::CPU)
	{
		#pragma omp parallel for
		for (int32_t i = 0; i < int32_t(length); ++i)
		{
			Color color = cumulativeImage.getPixel(i);
			color /= color.a;
			color.a = 1.0f;

			normalizedImage.setPixel(i, color);
		}
	}
	else
	{
#ifdef USE_CUDA

		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		normalizeKernel<<<dimGrid, dimBlock>>>(cumulativeImage.getSurfaceObject(), normalizedImage.getSurfaceObject(), width, height);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch normalize kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute normalize kernel");

#endif
	}
}

#ifdef USE_CUDA

__global__ void tonemapKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color;
	surf2Dread(&color, input, x * sizeof(float4), y);

	color.x = pow(color.x, 1.0f / 2.2f);
	color.y = pow(color.y, 1.0f / 2.2f);
	color.z = pow(color.z, 1.0f / 2.2f);
	color.w = 1.0f;

	surf2Dwrite(color, output, x * sizeof(float4), y);
}

#endif

void Film::tonemap(Tonemapper& tonemapper, RendererType type)
{
	if (type == RendererType::CPU)
	{
		tonemapper.apply(normalizedImage, tonemappedImage);
	}
	else
	{
#ifdef USE_CUDA

		dim3 dimBlock(16, 16);
		dim3 dimGrid;

		dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
		dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

		tonemapKernel<<<dimGrid, dimBlock>>>(normalizedImage.getSurfaceObject(), tonemappedImage.getSurfaceObject(), width, height);
		CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch tonemap kernel");
		CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute tonemap kernel");

#endif
	}
}

#ifdef USE_CUDA

__global__ void updateTextureKernel(cudaSurfaceObject_t input, cudaSurfaceObject_t output, uint32_t width, uint32_t height)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= width || y >= height)
		return;

	float4 color;
	surf2Dread(&color, input, x * sizeof(float4), y);
	surf2Dwrite(color, output, x * sizeof(float4), y);
}

#endif

void Film::updateTexture(RendererType type)
{
	if (windowed)
	{
		if (type == RendererType::CPU)
		{
			glBindTexture(GL_TEXTURE_2D, textureId);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(width), GLsizei(height), GL_RGBA, GL_FLOAT, tonemappedImage.getData());
			glBindTexture(GL_TEXTURE_2D, 0);

			GLUtils::checkError("Could not upload OpenGL texture data");
		}
		else
		{
#ifdef USE_CUDA

			CudaUtils::checkError(cudaGraphicsMapResources(1, &textureResource, 0), "Could not map texture resource");

			cudaArray_t textureData;
			CudaUtils::checkError(cudaGraphicsSubResourceGetMappedArray(&textureData, textureResource, 0, 0), "Could not get mapped array");

			cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = textureData;

			cudaSurfaceObject_t surfaceObject;
			CudaUtils::checkError(cudaCreateSurfaceObject(&surfaceObject, &resDesc), "Could not create surface object");
			
			dim3 dimBlock(16, 16);
			dim3 dimGrid;

			dimGrid.x = (width + dimBlock.x - 1) / dimBlock.x;
			dimGrid.y = (height + dimBlock.y - 1) / dimBlock.y;

			updateTextureKernel<<<dimGrid, dimBlock>>>(tonemappedImage.getSurfaceObject(), surfaceObject, width, height);
			CudaUtils::checkError(cudaPeekAtLastError(), "Could not launch update texture kernel");
			CudaUtils::checkError(cudaDeviceSynchronize(), "Could not execute update texture kernel");

			CudaUtils::checkError(cudaDestroySurfaceObject(surfaceObject), "Could not destroy surface object");
			CudaUtils::checkError(cudaGraphicsUnmapResources(1, &textureResource, 0), "Could not unmap texture resource");

#endif
		}
	}
}

Color Film::getCumulativeColor(uint32_t x, uint32_t y) const
{
	return cumulativeImage.getPixel(x, y);
}

Color Film::getNormalizedColor(uint32_t x, uint32_t y) const
{
	return normalizedImage.getPixel(x, y);
}

Color Film::getTonemappedColor(uint32_t x, uint32_t y) const
{
	return tonemappedImage.getPixel(x, y);
}

CUDA_CALLABLE Image& Film::getCumulativeImage()
{
	return cumulativeImage;
}

CUDA_CALLABLE Image& Film::getNormalizedImage()
{
	return normalizedImage;
}

CUDA_CALLABLE Image& Film::getTonemappedImage()
{
	return tonemappedImage;
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

GLuint Film::getTextureId() const
{
	return textureId;
}

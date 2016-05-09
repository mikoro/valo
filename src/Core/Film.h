// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <atomic>
#include <cstdint>

#include <GL/glcorearb.h>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "Core/Common.h"
#include "Core/Image.h"

namespace Valo
{
	class Color;
	class Tonemapper;

	class Film
	{
	public:

		explicit Film(bool windowed);

		void initialize();
		void shutdown();
		void resize(uint32_t width, uint32_t height, RendererType type);
		void clear(RendererType type);
		bool hasBeenCleared() const;
		void resetCleared();
		void load(uint32_t width, uint32_t height, const std::string& fileName, RendererType type);
		void loadMultiple(uint32_t width, uint32_t height, const std::string& dirName, RendererType type);
		void save(const std::string& fileName, bool writeToLog = true) const;

		CUDA_CALLABLE void addSample(uint32_t x, uint32_t y, const Color& color, float filterWeight);
		CUDA_CALLABLE void addSample(uint32_t index, const Color& color, float filterWeight);

		void normalize(RendererType type);
		void tonemap(Tonemapper& tonemapper, RendererType type);
		void updateTexture(RendererType type);

		Color getCumulativeColor(uint32_t x, uint32_t y) const;
		Color getNormalizedColor(uint32_t x, uint32_t y) const;
		Color getTonemappedColor(uint32_t x, uint32_t y) const;

		CUDA_CALLABLE Image& getCumulativeImage();
		CUDA_CALLABLE Image& getNormalizedImage();
		CUDA_CALLABLE Image& getTonemappedImage();

		CUDA_CALLABLE uint32_t getWidth() const;
		CUDA_CALLABLE uint32_t getHeight() const;
		CUDA_CALLABLE uint32_t getLength() const;

		GLuint getTextureId() const;

		std::atomic<uint32_t> pixelSamples;

	private:

		uint32_t width = 0;
		uint32_t height = 0;
		uint32_t length = 0;

		bool windowed = false;
		bool cleared = false;
		
		Image cumulativeImage;
		Image normalizedImage;
		Image tonemappedImage;

		GLuint textureId = 0;

#ifdef USE_CUDA
		cudaGraphicsResource* textureResource = nullptr;
#endif
	};
}

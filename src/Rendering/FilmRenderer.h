// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <GL/glcorearb.h>

#include "Rendering/Image.h"

namespace Raycer
{
	class Film;

	class FilmRenderer
	{
	public:

		void initialize();
		void setFilmSize(uint64_t width, uint64_t height);
		void setWindowSize(uint64_t width, uint64_t height);
		void uploadFilmData(const Film& film);
		void render();

		GLuint getFilmTextureId() const;

	private:

		Imagef filmData;

		uint64_t filmWidth = 0;
		uint64_t filmHeight = 0;
		uint64_t windowWidth = 0;
		uint64_t windowHeight = 0;

		GLuint vaoId = 0;
		GLuint vboId = 0;
		GLuint filmTextureId = 0;

		GLuint resampleProgramId = 0;
		GLuint resampleTextureUniformId = 0;
		GLuint resampleTextureWidthUniformId = 0;
		GLuint resampleTextureHeightUniformId = 0;
		GLuint resampleTexelWidthUniformId = 0;
		GLuint resampleTexelHeightUniformId = 0;
	};
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <GL/glcorearb.h>

namespace Raycer
{
	class Film;

	class FilmQuad
	{
	public:

		void initialize();
		void resize(uint32_t width, uint32_t height);
		void upload(const Film& film);
		void render();

	private:

		uint32_t width = 0;
		uint32_t height = 0;

		GLuint vaoId = 0;
		GLuint vboId = 0;
		GLuint textureId = 0;

		GLuint programId = 0;
		GLuint textureUniformId = 0;
		GLuint textureWidthUniformId = 0;
		GLuint textureHeightUniformId = 0;
		GLuint texelWidthUniformId = 0;
		GLuint texelHeightUniformId = 0;
	};
}

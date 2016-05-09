// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <GL/glcorearb.h>

namespace Valo
{
	class Film;

	class FilmQuad
	{
	public:

		void initialize();
		void render(const Film& film);

	private:

		GLuint vaoId = 0;
		GLuint vboId = 0;

		GLuint programId = 0;
		GLuint textureUniformId = 0;
		GLuint textureWidthUniformId = 0;
		GLuint textureHeightUniformId = 0;
		GLuint texelWidthUniformId = 0;
		GLuint texelHeightUniformId = 0;
	};
}

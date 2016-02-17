// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Rendering/FilmRenderer.h"
#include "App.h"
#include "Utils/Settings.h"
#include "Utils/GLHelper.h"
#include "Film.h"

using namespace Raycer;

void FilmRenderer::initialize()
{
	Settings& settings = App::getSettings();

	glGenTextures(1, &filmTextureId);

	GLHelper::checkError("Could not create OpenGL textures");

	glBindTexture(GL_TEXTURE_2D, filmTextureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	GLHelper::checkError("Could not set OpenGL texture parameters");

	resampleProgramId = GLHelper::buildProgram("data/shaders/film.vert", "data/shaders/film.frag");

	resampleTextureUniformId = glGetUniformLocation(resampleProgramId, "tex0");
	resampleTextureWidthUniformId = glGetUniformLocation(resampleProgramId, "textureWidth");
	resampleTextureHeightUniformId = glGetUniformLocation(resampleProgramId, "textureHeight");
	resampleTexelWidthUniformId = glGetUniformLocation(resampleProgramId, "texelWidth");
	resampleTexelHeightUniformId = glGetUniformLocation(resampleProgramId, "texelHeight");

	GLHelper::checkError("Could not get GLSL uniforms");

	const GLfloat vertexData[] =
	{
		-1.0f, -1.0f, 0.0f, 0.0f,
		1.0f, -1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 1.0f, 1.0f,
		-1.0f, 1.0f, 0.0f, 1.0f,
	};

	glGenBuffers(1, &vboId);
	glBindBuffer(GL_ARRAY_BUFFER, vboId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexData), vertexData, GL_STATIC_DRAW);

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), nullptr);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), reinterpret_cast<void*>(2 * sizeof(GLfloat)));
	glBindVertexArray(0);

	GLHelper::checkError("Could not set OpenGL buffer parameters");

	setWindowSize(settings.window.width, settings.window.height);
}

void FilmRenderer::setFilmSize(uint64_t width, uint64_t height)
{
	filmWidth = width;
	filmHeight = height;

	// reserve the texture memory on the device
	glBindTexture(GL_TEXTURE_2D, filmTextureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GLsizei(filmWidth), GLsizei(filmHeight), 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLHelper::checkError("Could not reserve OpenGL texture memory");
}

void FilmRenderer::setWindowSize(uint64_t width, uint64_t height)
{
	windowWidth = width;
	windowHeight = height;
}

void FilmRenderer::uploadFilmData(const Film& film)
{
	glBindTexture(GL_TEXTURE_2D, filmTextureId);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(filmWidth), GLsizei(filmHeight), GL_RGBA, GL_FLOAT, &film.getOutputImage().getPixelDataConst()[0]);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLHelper::checkError("Could not upload OpenGL texture data");
}

void FilmRenderer::render()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, filmTextureId);

	glUseProgram(resampleProgramId);
	glUniform1i(resampleTextureUniformId, 0);
	glUniform1f(resampleTextureWidthUniformId, float(filmWidth));
	glUniform1f(resampleTextureHeightUniformId, float(filmHeight));
	glUniform1f(resampleTexelWidthUniformId, 1.0f / filmWidth);
	glUniform1f(resampleTexelHeightUniformId, 1.0f / filmHeight);

	glBindVertexArray(vaoId);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

	GLHelper::checkError("Could not render the film");
}

GLuint FilmRenderer::getFilmTextureId() const
{
	return filmTextureId;
}

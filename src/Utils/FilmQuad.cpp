// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Film.h"
#include "Utils/FilmQuad.h"
#include "Utils/GLHelper.h"

using namespace Raycer;

void FilmQuad::initialize()
{
	glGenTextures(1, &textureId);

	GLHelper::checkError("Could not create OpenGL texture");

	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	GLHelper::checkError("Could not set OpenGL texture parameters");

	programId = GLHelper::buildProgram("data/shaders/film.vert", "data/shaders/film.frag");

	textureUniformId = glGetUniformLocation(programId, "tex0");
	textureWidthUniformId = glGetUniformLocation(programId, "textureWidth");
	textureHeightUniformId = glGetUniformLocation(programId, "textureHeight");
	texelWidthUniformId = glGetUniformLocation(programId, "texelWidth");
	texelHeightUniformId = glGetUniformLocation(programId, "texelHeight");

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
}

void FilmQuad::resize(uint64_t width_, uint64_t height_)
{
	width = width_;
	height = height_;

	// reserve the texture memory on the device
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, GLsizei(width), GLsizei(height), 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	GLHelper::checkError("Could not reserve OpenGL texture memory");
}

void FilmQuad::upload(const Film& film)
{
	glBindTexture(GL_TEXTURE_2D, textureId);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(width), GLsizei(height), GL_RGBA, GL_FLOAT, film.getImage().getPixelData());
	glBindTexture(GL_TEXTURE_2D, 0);

	GLHelper::checkError("Could not upload OpenGL texture data");
}

void FilmQuad::render()
{
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureId);

	glUseProgram(programId);
	glUniform1i(textureUniformId, 0);
	glUniform1f(textureWidthUniformId, float(width));
	glUniform1f(textureHeightUniformId, float(height));
	glUniform1f(texelWidthUniformId, 1.0f / width);
	glUniform1f(texelHeightUniformId, 1.0f / height);

	glBindVertexArray(vaoId);
	glDrawArrays(GL_TRIANGLES, 0, 6);

	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);

	GLHelper::checkError("Could not render the film");
}

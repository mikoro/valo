// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Common.h"
#include "Core/Film.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Renderers/CudaRenderer.h"
#include "Renderers/Renderer.h"

using namespace Raycer;

void CudaRenderer::initialize()
{
}

#ifdef USE_CUDA

__global__ void renderKernel(Scene& scene, Film& film, bool filtering)
{
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= film.getWidth() || y >= film.getHeight())
		return;

	Vector2 pixel = Vector2(x, y);
	Random random;
	float filterWeight = 1.0f;

	if (filtering && scene.renderer.filtering)
	{
		Vector2 offset = (random.getVector2() - Vector2(0.5f, 0.5f)) * 2.0f * scene.renderer.filter.getRadius();
		filterWeight = scene.renderer.filter.getWeight(offset);
		pixel += offset;
	}

	bool isOffLens;
	Ray viewRay = scene.camera.getRay(pixel, isOffLens);

	if (isOffLens)
	{
		film.addSample(x, y, scene.general.offLensColor, filterWeight);
		return;
	}

	Color color = scene.integrator.calculateRadiance(scene, viewRay, random);
	film.addSample(x, y, color, filterWeight);
}

void CudaRenderer::render(RenderJob& job, bool filtering)
{
	Scene& scene = *job.scene;
	Film& film = *job.film;

	dim3 dimBlock(16, 16);
	dim3 dimGrid;

	dimGrid.x = (film.getWidth() + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (film.getHeight() + dimBlock.y - 1) / dimBlock.y;

	renderKernel<<<dimGrid, dimBlock>>>(scene, film, filtering);
	cudaDeviceSynchronize();
}

#else

void CudaRenderer::render(RenderJob& job, bool filtering)
{
	job.film->clear();
	(void)filtering;
}

#endif

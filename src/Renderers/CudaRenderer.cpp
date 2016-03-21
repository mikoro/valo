// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Film.h"
#include "Core/Scene.h"
#include "Renderers/CudaRenderer.h"
#include "Renderers/Renderer.h"

using namespace Raycer;

void CudaRenderer::initialize()
{
}

void CudaRenderer::render(RenderJob& job, bool filtering)
{
	//Scene& scene = *job.scene;
	Film& film = *job.film;
	(void)filtering;
	
	film.clear();
}

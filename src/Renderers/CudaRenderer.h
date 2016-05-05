// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Utils/CudaAlloc.h"

namespace Raycer
{
	struct RenderJob;
	class Scene;
	class Film;
	struct RandomGeneratorState;

	class CudaRenderer
	{
	public:

		CudaRenderer();

		void initialize();
		void resize(uint32_t width, uint32_t height);
		void render(RenderJob& job, bool filtering);

	private:

		CudaAlloc<Scene> sceneAlloc;
		CudaAlloc<Film> filmAlloc;
		CudaAlloc<RandomGeneratorState> randomStatesAlloc;
	};
}

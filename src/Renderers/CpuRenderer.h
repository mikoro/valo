// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Math/Random.h"

namespace Valo
{
	struct RenderJob;
	
	class CpuRenderer
	{
	public:

		void initialize();
		void resize(uint32_t width, uint32_t height);
		void render(RenderJob& job, bool filtering);

		int32_t maxThreadCount = 4;

	private:

		std::vector<Random> randoms;
	};
}

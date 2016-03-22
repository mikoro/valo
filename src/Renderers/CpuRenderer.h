// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/Random.h"

namespace Raycer
{
	struct RenderJob;
	
	class CpuRenderer
	{
	public:

		void initialize();
		void render(RenderJob& job, bool filtering);

		int32_t maxThreadCount = 4;

	private:

		std::vector<Random> randoms;
	};
}

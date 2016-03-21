// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	struct RenderJob;

	class CudaRenderer
	{
	public:

		void initialize();
		void render(RenderJob& job, bool filtering);
	};
}

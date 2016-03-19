// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	struct RenderJob;

	class CudaRenderer
	{
	public:

		void initialize();
		void render(RenderJob& job);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
		}
	};
}

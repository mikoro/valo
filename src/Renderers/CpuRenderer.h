// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Utils/Random.h"

namespace Raycer
{
	struct RenderJob;
	
	class CpuRenderer
	{
	public:

		void initialize();
		void render(RenderJob& job);

		int32_t maxThreadCount = 4;

	private:

		std::vector<Random> randoms;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(maxThreadCount));
		}
	};
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Utils/PerlinNoise.h"
#include "Utils/ColorGradient.h"
#include "Math/Color.h"

namespace Valo
{
	class Vector2;
	class Vector3;

	class FireTexture
	{
	public:

		void initialize();

		CUDA_CALLABLE Color getColor(const Vector2& texcoord, const Vector3& position) const;
		
		uint32_t seed = 1;
		float scale = 1.0f;

	private:

		PerlinNoise noise;
		ColorGradient gradient;
	};
}

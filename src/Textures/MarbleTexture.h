// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Utils/PerlinNoise.h"
#include "Math/Color.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class MarbleTexture
	{
	public:

		void initialize();

		CUDA_CALLABLE Color getColor(const Vector2& texcoord, const Vector3& position) const;
		
		uint32_t seed = 1;
		Color marbleColor = Color(255, 252, 240);
		Color streakColor = Color(0, 33, 71);
		float density = 10.0f;
		float swirliness = 15.0f;
		float transparency = 2.0f;

	private:

		PerlinNoise noise;
	};
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Math/Color.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class CheckerTexture
	{
	public:

		void initialize();

		CUDA_CALLABLE Color getColor(const Vector2& texcoord, const Vector3& position);
		
		Color color1;
		Color color2;
		bool stripeMode = false;
		float stripeWidth = 0.05f;
	};
}

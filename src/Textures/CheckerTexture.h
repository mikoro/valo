// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Math/Color.h"

namespace Valo
{
	class Vector2;
	class Vector3;

	class CheckerTexture
	{
	public:

		CUDA_CALLABLE Color getColor(const Vector2& texcoord, const Vector3& position) const;
		
		Color color1 = Color(0, 0, 0);
		Color color2 = Color(255, 255, 255);
		bool stripeMode = false;
		float stripeWidth = 0.05f;
	};
}

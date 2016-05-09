// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Math/Vector2.h"

namespace Valo
{
	class Vector2;

	class GaussianFilter
	{
	public:

		CUDA_CALLABLE float getWeight(float s) const;
		CUDA_CALLABLE float getWeight(const Vector2& point) const;

		CUDA_CALLABLE Vector2 getRadius() const;

		Vector2 stdDeviation = Vector2(0.5f, 0.5f);
	};
}

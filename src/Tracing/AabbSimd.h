// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/Vector3.h"

namespace Raycer
{
	class Aabb;

	class AabbSimd
	{
	public:

		AabbSimd();
		explicit AabbSimd(const Aabb& aabb);

		void expand(const AabbSimd& other);
		float getSurfaceArea() const;
		Aabb getAabb() const;

		__m128 min;
		__m128 max;
	};
}

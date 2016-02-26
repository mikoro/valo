// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <limits>

#include "Math/Vector3.h"

namespace Raycer
{
	class Ray
	{
	public:

		void precalculate();

		Vector3 origin;
		Vector3 direction;
		Vector3 inverseDirection;

		float minDistance = 0.0f;
		float maxDistance = std::numeric_limits<float>::max();

		bool isShadowRay = false;
		bool fastOcclusion = false;
		bool directionIsNegative[3];
	};
}

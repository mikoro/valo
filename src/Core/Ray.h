// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cfloat>

#include "Core/Common.h"
#include "Math/Vector3.h"

namespace Raycer
{
	class Ray
	{
	public:

		CUDA_CALLABLE void precalculate();

		Vector3 origin;
		Vector3 direction;
		Vector3 inverseDirection;

		float minDistance = 0.0f;
		float maxDistance = FLT_MAX;

		bool isVisibilityRay = false;
		bool directionIsNegative[3];
	};
}

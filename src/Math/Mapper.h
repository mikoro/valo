// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/ONB.h"

#include "Core/Common.h"

/*

point is assumed to be in [0, 1]^2

*/

namespace Raycer
{
	class Vector2;
	class Vector3;

	class Mapper
	{
	public:

		CUDA_CALLABLE static Vector2 mapToDisc(const Vector2& point);
		CUDA_CALLABLE static Vector3 mapToCosineHemisphere(const Vector2& point, const ONB& onb = ONB::up());
		CUDA_CALLABLE static Vector3 mapToUniformHemisphere(const Vector2& point, const ONB& onb = ONB::up());
	};
}

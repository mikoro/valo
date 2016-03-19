// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/ONB.h"

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

		static Vector2 mapToDisc(const Vector2& point);
		static Vector3 mapToCosineHemisphere(const Vector2& point, const ONB& onb = ONB::UP);
		static Vector3 mapToUniformHemisphere(const Vector2& point, const ONB& onb = ONB::UP);
	};
}

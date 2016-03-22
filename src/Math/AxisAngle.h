// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Math/Vector3.h"

namespace Raycer
{
	class Matrix4x4;

	class AxisAngle
	{
	public:

		CUDA_CALLABLE explicit AxisAngle(const Vector3& axis = Vector3::up(), float angle = 0.0f);

		CUDA_CALLABLE Matrix4x4 toMatrix4x4() const;

		Vector3 axis;
		float angle; // degrees
	};
}

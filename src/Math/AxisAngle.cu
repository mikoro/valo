// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/AxisAngle.h"
#include "Math/Matrix4x4.h"
#include "Math/MathUtils.h"

using namespace Raycer;

CUDA_CALLABLE AxisAngle::AxisAngle(const Vector3& axis_, float angle_) : axis(axis_), angle(angle_)
{
}

CUDA_CALLABLE Matrix4x4 AxisAngle::toMatrix4x4() const
{
	float x = axis.x;
	float y = axis.y;
	float z = axis.z;
	float c = std::cos(MathUtils::degToRad(angle));
	float s = std::sin(MathUtils::degToRad(angle));
	float ci = 1.0f - c;

	Matrix4x4 result(
		c + x * x * ci, x * y * ci - z * s, x * z * ci + y * s, 0.0f,
		y * x * ci + z * s, c + y * y * ci, y * z * ci - x * s, 0.0f,
		z * x * ci - y * s, z * y * ci + x * s, c + z * z * ci, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	return result;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

/*

http://3dgep.com/understanding-quaternions/
q = w + xi + yj + zk

*/

namespace Raycer
{
	class AxisAngle;
	class Matrix4x4;
	class Vector3;

	class Quaternion
	{
	public:

		CUDA_CALLABLE explicit Quaternion(float w = 0.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f);
		CUDA_CALLABLE explicit Quaternion(const AxisAngle& axisAngle);
		CUDA_CALLABLE Quaternion(const Vector3& axis, float angle);

		CUDA_CALLABLE friend Quaternion operator+(const Quaternion& q1, const Quaternion& q2);
		CUDA_CALLABLE friend Quaternion operator-(const Quaternion& q1, const Quaternion& q2);
		CUDA_CALLABLE friend Quaternion operator*(const Quaternion& q1, const Quaternion& q2);
		CUDA_CALLABLE friend Quaternion operator*(const Quaternion& q, float s);
		CUDA_CALLABLE friend Quaternion operator*(float s, const Quaternion& q);
		CUDA_CALLABLE friend Quaternion operator/(const Quaternion& q, float s);
		CUDA_CALLABLE friend Quaternion operator-(const Quaternion& q);

		CUDA_CALLABLE friend bool operator==(const Quaternion& q1, const Quaternion& q2);
		CUDA_CALLABLE friend bool operator!=(const Quaternion& q1, const Quaternion& q2);

		CUDA_CALLABLE Quaternion& operator+=(const Quaternion& q);
		CUDA_CALLABLE Quaternion& operator-=(const Quaternion& q);
		CUDA_CALLABLE Quaternion& operator*=(const Quaternion& q);
		CUDA_CALLABLE Quaternion& operator*=(float s);
		CUDA_CALLABLE Quaternion& operator/=(float s);

		CUDA_CALLABLE Vector3 rotate(const Vector3& v) const;
		CUDA_CALLABLE float length() const;
		CUDA_CALLABLE float lengthSquared() const;
		CUDA_CALLABLE void conjugate();
		CUDA_CALLABLE Quaternion conjugated() const;
		CUDA_CALLABLE void normalize();
		CUDA_CALLABLE Quaternion normalized() const;
		CUDA_CALLABLE bool isZero() const;
		bool isNan() const;
		CUDA_CALLABLE bool isNormal() const;
		CUDA_CALLABLE float dot(const Quaternion& q) const;
		CUDA_CALLABLE AxisAngle toAxisAngle() const;
		CUDA_CALLABLE Matrix4x4 toMatrix4x4() const;

		CUDA_CALLABLE static Quaternion lerp(const Quaternion& q1, const Quaternion& q2, float t);
		CUDA_CALLABLE static Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t);

		float w;
		float x;
		float y;
		float z;
	};
}

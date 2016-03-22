// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cassert>

#include "Math/Quaternion.h"
#include "Math/AxisAngle.h"
#include "Math/Matrix4x4.h"
#include "Math/Vector3.h"
#include "Math/MathUtils.h"

using namespace Raycer;

CUDA_CALLABLE Quaternion::Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_)
{
}

CUDA_CALLABLE Quaternion::Quaternion(const AxisAngle& axisAngle) : Quaternion(axisAngle.axis, axisAngle.angle)
{
}

CUDA_CALLABLE Quaternion::Quaternion(const Vector3& axis, float angle)
{
	assert(axis.isNormal());

	float radiansPer2 = MathUtils::degToRad(angle) / 2.0f;
	float cosine = std::cos(radiansPer2);
	float sine = std::sin(radiansPer2);

	w = cosine;
	x = axis.x * sine;
	y = axis.y * sine;
	z = axis.z * sine;
}

namespace Raycer
{
	CUDA_CALLABLE Quaternion operator+(const Quaternion& q1, const Quaternion& q2)
	{
		return Quaternion(q1.w + q2.w, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z);
	}

	CUDA_CALLABLE Quaternion operator-(const Quaternion& q1, const Quaternion& q2)
	{
		return Quaternion(q1.w - q2.w, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z);
	}

	CUDA_CALLABLE Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
	{
		Quaternion r;

		r.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
		r.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
		r.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
		r.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

		return r;
	}

	CUDA_CALLABLE Quaternion operator*(const Quaternion& q, float s)
	{
		return Quaternion(q.w * s, q.x * s, q.y * s, q.z * s);
	}

	CUDA_CALLABLE Quaternion operator*(float s, const Quaternion& q)
	{
		return q * s;
	}

	CUDA_CALLABLE Quaternion operator/(const Quaternion& q, float s)
	{
		float invS = 1.0f / s;
		return Quaternion(q.w * invS, q.x * invS, q.y * invS, q.z * invS);
	}

	CUDA_CALLABLE Quaternion operator-(const Quaternion& q)
	{
		return Quaternion(-q.w, -q.x, -q.y, -q.z);
	}

	CUDA_CALLABLE bool operator==(const Quaternion& q1, const Quaternion& q2)
	{
		return MathUtils::almostSame(q1.w, q2.w) && MathUtils::almostSame(q1.x, q2.x) && MathUtils::almostSame(q1.y, q2.y) && MathUtils::almostSame(q1.z, q2.z);
	}

	CUDA_CALLABLE bool operator!=(const Quaternion& q1, const Quaternion& q2)
	{
		return !(q1 == q2);
	}
}

CUDA_CALLABLE Quaternion& Quaternion::operator+=(const Quaternion& q)
{
	*this = *this + q;
	return *this;
}

CUDA_CALLABLE Quaternion& Quaternion::operator-=(const Quaternion& q)
{
	*this = *this - q;
	return *this;
}

CUDA_CALLABLE Quaternion& Quaternion::operator*=(const Quaternion& q)
{
	*this = *this * q;
	return *this;
}

CUDA_CALLABLE Quaternion& Quaternion::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

CUDA_CALLABLE Quaternion& Quaternion::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

CUDA_CALLABLE Vector3 Quaternion::rotate(const Vector3& v) const
{
	Vector3 r(x, y, z);
	return v + 2.0f * r.cross(r.cross(v) + w * v);
}

CUDA_CALLABLE float Quaternion::length() const
{
	return std::sqrt(w * w + x * x + y * y + z * z);
}

CUDA_CALLABLE float Quaternion::lengthSquared() const
{
	return (w * w + x * x + y * y + z * z);
}

CUDA_CALLABLE void Quaternion::conjugate()
{
	x = -x;
	y = -y;
	z = -z;
}

CUDA_CALLABLE Quaternion Quaternion::conjugated() const
{
	return Quaternion(w, -x, -y, -z);
}

CUDA_CALLABLE void Quaternion::normalize()
{
	*this /= length();
}

CUDA_CALLABLE Quaternion Quaternion::normalized() const
{
	return *this / length();
}

CUDA_CALLABLE bool Quaternion::isZero() const
{
	return (w == 0.0f && x == 0.0f && y == 0.0f && z == 0.0f);
}

bool Quaternion::isNan() const
{
	return (std::isnan(w) || std::isnan(x) || std::isnan(y) || std::isnan(z));
}

CUDA_CALLABLE bool Quaternion::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

CUDA_CALLABLE float Quaternion::dot(const Quaternion& q) const
{
	return (w * q.w) + (x * q.x) + (y * q.y) + (z * q.z);
}

CUDA_CALLABLE AxisAngle Quaternion::toAxisAngle() const
{
	assert(isNormal());

	AxisAngle aa;

	aa.angle = MathUtils::radToDeg(2.0f * acos(w));
	float sine2 = 1.0f - w * w;

	if (sine2 < FLT_EPSILON)
	{
		aa.axis.x = 1.0f;
		aa.axis.y = 0.0f;
		aa.axis.z = 0.0f;
	}
	else
	{
		float invSine = 1.0f / sqrt(sine2);

		aa.axis.x = x * invSine;
		aa.axis.y = y * invSine;
		aa.axis.z = z * invSine;
	}

	return aa;
}

CUDA_CALLABLE Matrix4x4 Quaternion::toMatrix4x4() const
{
	assert(isNormal());

	Matrix4x4 result(
		1.0f - 2.0f * y * y - 2.0f * z * z, 2.0f * x * y - 2.0f * w * z, 2.0f * x * z + 2.0f * w * y, 0.0f,
		2.0f * x * y + 2.0f * w * z, 1.0f - 2.0f * x * x - 2.0f * z * z, 2.0f * y * z + 2.0f * w * x, 0.0f,
		2.0f * x * z - 2.0f * w * y, 2.0f * y * z - 2.0f * w * x, 1.0f - 2.0f * x * x - 2.0f * y * y, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	return result;
}

CUDA_CALLABLE Quaternion Quaternion::lerp(const Quaternion& q1, const Quaternion& q2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return q1 * (1.0f - t) + q2 * t;
}

CUDA_CALLABLE Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);

	float cosine = q1.dot(q2) / (q1.length() * q2.length());
	float theta = acos(cosine);
	float sinTheta = sin(theta);

	if (sinTheta < FLT_EPSILON)
		return lerp(q1, q2, t);
	
	Quaternion tq1 = (cosine < 0.0f) ? -q1 : q1;
	Quaternion q = sin((1.0f - t) * theta) * tq1 + sin(t * theta) * q2;

	return q / sinTheta;
}

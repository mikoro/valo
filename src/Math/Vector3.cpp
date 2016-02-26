// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/Vector3.h"
#include "Math/Vector4.h"
#include "Math/MathUtils.h"

using namespace Raycer;

const Vector3 Vector3::RIGHT = Vector3(1.0f, 0.0f, 0.0f);
const Vector3 Vector3::UP = Vector3(0.0f, 1.0f, 0.0f);
const Vector3 Vector3::FORWARD = Vector3(0.0f, 0.0f, 1.0f); // out of the screen, towards the viewer
const Vector3 Vector3::ALMOST_UP = Vector3(0.000001f, 1.0f, 0.000001f);

Vector3::Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_)
{
}

Vector3::Vector3(const Vector4& v) : x(v.x), y(v.y), z(v.z)
{
}

namespace Raycer
{
	Vector3 operator+(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x + w.x, v.y + w.y, v.z + w.z);
	}

	Vector3 operator-(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x - w.x, v.y - w.y, v.z - w.z);
	}

	Vector3 operator*(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x * w.x, v.y * w.y, v.z * w.z);
	}

	Vector3 operator*(const Vector3& v, float s)
	{
		return Vector3(v.x * s, v.y * s, v.z * s);
	}

	Vector3 operator*(float s, const Vector3& v)
	{
		return Vector3(v.x * s, v.y * s, v.z * s);
	}

	Vector3 operator/(const Vector3& v, const Vector3& w)
	{
		return Vector3(v.x / w.x, v.y / w.y, v.z / w.z);
	}

	Vector3 operator/(const Vector3& v, float s)
	{
		float invS = 1.0f / s;
		return Vector3(v.x * invS, v.y * invS, v.z * invS);
	}

	Vector3 operator-(const Vector3& v)
	{
		return Vector3(-v.x, -v.y, -v.z);
	}

	bool operator==(const Vector3& v, const Vector3& w)
	{
		return MathUtils::almostSame(v.x, w.x) && MathUtils::almostSame(v.y, w.y) && MathUtils::almostSame(v.z, w.z);
	}

	bool operator!=(const Vector3& v, const Vector3& w)
	{
		return !(v == w);
	}

	bool operator>(const Vector3& v, const Vector3& w)
	{
		return v.x > w.x && v.y > w.y && v.z > w.z;
	}

	bool operator<(const Vector3& v, const Vector3& w)
	{
		return v.x < w.x && v.y < w.y && v.z < w.z;
	}
}

Vector3& Vector3::operator+=(const Vector3& v)
{
	*this = *this + v;
	return *this;
}

Vector3& Vector3::operator-=(const Vector3& v)
{
	*this = *this - v;
	return *this;
}

Vector3& Vector3::operator*=(const Vector3& v)
{
	*this = *this * v;
	return *this;
}

Vector3& Vector3::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

Vector3& Vector3::operator/=(const Vector3& v)
{
	*this = *this / v;
	return *this;
}

Vector3& Vector3::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

float Vector3::operator[](uint64_t index) const
{
	return (&x)[index];
}

float Vector3::getElement(uint64_t index) const
{
	switch (index)
	{
		case 0: return x;
		case 1: return y;
		case 2: return z;
		default: throw std::runtime_error("Invalid vector element index");
	}
}

void Vector3::setElement(uint64_t index, float value)
{
	switch (index)
	{
		case 0: x = value;
		case 1: y = value;
		case 2: z = value;
		default: throw std::runtime_error("Invalid vector element index");
	}
}

float Vector3::length() const
{
	return std::sqrt(x * x + y * y + z * z);
}

float Vector3::lengthSquared() const
{
	return (x * x + y * y + z * z);
}

void Vector3::normalize()
{
	*this /= length();
}

Vector3 Vector3::normalized() const
{
	return *this / length();
}

void Vector3::inverse()
{
	x = 1.0f / x;
	y = 1.0f / y;
	z = 1.0f / z;
}

Vector3 Vector3::inversed() const
{
	Vector3 inverse;

	inverse.x = 1.0f / x;
	inverse.y = 1.0f / y;
	inverse.z = 1.0f / z;

	return inverse;
}

bool Vector3::isZero() const
{
	return (x == 0.0f && y == 0.0f && z == 0.0f);
}

bool Vector3::isNan() const
{
	return (std::isnan(x) || std::isnan(y) || std::isnan(z));
}

bool Vector3::isNormal() const
{
	return MathUtils::almostSame(lengthSquared(), 1.0f);
}

float Vector3::dot(const Vector3& v) const
{
	return (x * v.x) + (y * v.y) + (z * v.z);
}

Vector3 Vector3::cross(const Vector3& v) const
{
	Vector3 r;

	r.x = y * v.z - z * v.y;
	r.y = z * v.x - x * v.z;
	r.z = x * v.y - y * v.x;

	return r;
}

Vector3 Vector3::reflect(const Vector3& normal) const
{
	return *this - ((2.0f * this->dot(normal)) * normal);
}

std::string Vector3::toString() const
{
	return tfm::format("(%.2f, %.2f, %.2f)", x, y, z);
}

Vector4 Vector3::toVector4(float w_) const
{
	return Vector4(x, y, z, w_);
}

Vector3 Vector3::lerp(const Vector3& v1, const Vector3& v2, float t)
{
	assert(t >= 0.0f && t <= 1.0f);
	return v1 * (1.0f - t) + v2 * t;
}

Vector3 Vector3::abs(const Vector3& v)
{
	return Vector3(std::abs(v.x), std::abs(v.y), std::abs(v.z));
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Vector4;

	class Vector3
	{
	public:

		explicit Vector3(float x = 0.0f, float y = 0.0f, float z = 0.0f);
		explicit Vector3(const Vector4& v);

		friend Vector3 operator+(const Vector3& v, const Vector3& w);
		friend Vector3 operator-(const Vector3& v, const Vector3& w);
		friend Vector3 operator*(const Vector3& v, const Vector3& w);
		friend Vector3 operator*(const Vector3& v, float s);
		friend Vector3 operator*(float s, const Vector3& v);
		friend Vector3 operator/(const Vector3& v, const Vector3& w);
		friend Vector3 operator/(const Vector3& v, float s);
		friend Vector3 operator-(const Vector3& v);

		friend bool operator==(const Vector3& v, const Vector3& w);
		friend bool operator!=(const Vector3& v, const Vector3& w);
		friend bool operator>(const Vector3& v, const Vector3& w);
		friend bool operator<(const Vector3& v, const Vector3& w);

		Vector3& operator+=(const Vector3& v);
		Vector3& operator-=(const Vector3& v);
		Vector3& operator*=(const Vector3& v);
		Vector3& operator*=(float s);
		Vector3& operator/=(const Vector3& v);
		Vector3& operator/=(float s);
		float operator[](uint32_t index) const;

		float getElement(uint32_t index) const;
		void setElement(uint32_t index, float value);
		float length() const;
		float lengthSquared() const;
		void normalize();
		Vector3 normalized() const;
		void inverse();
		Vector3 inversed() const;
		bool isZero() const;
		bool isNan() const;
		bool isNormal() const;
		float dot(const Vector3& v) const;
		Vector3 cross(const Vector3& v) const;
		Vector3 reflect(const Vector3& normal) const;
		std::string toString() const;
		Vector4 toVector4(float w = 0.0f) const;

		static Vector3 lerp(const Vector3& v1, const Vector3& v2, float t);
		static Vector3 abs(const Vector3& v);

		static const Vector3 RIGHT;
		static const Vector3 UP;
		static const Vector3 FORWARD;
		static const Vector3 ALMOST_UP;

		float x;
		float y;
		float z;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(x),
				CEREAL_NVP(y),
				CEREAL_NVP(z));
		}
	};
}

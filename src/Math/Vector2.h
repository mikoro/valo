// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Vector2
	{
	public:

		explicit Vector2(float x = 0.0f, float y = 0.0f);

		friend Vector2 operator+(const Vector2& v, const Vector2& w);
		friend Vector2 operator-(const Vector2& v, const Vector2& w);
		friend Vector2 operator*(const Vector2& v, const Vector2& w);
		friend Vector2 operator*(const Vector2& v, float s);
		friend Vector2 operator*(float s, const Vector2& v);
		friend Vector2 operator/(const Vector2& v, const Vector2& w);
		friend Vector2 operator/(const Vector2& v, float s);
		friend Vector2 operator-(const Vector2& v);

		friend bool operator==(const Vector2& v, const Vector2& w);
		friend bool operator!=(const Vector2& v, const Vector2& w);
		friend bool operator>(const Vector2& v, const Vector2& w);
		friend bool operator<(const Vector2& v, const Vector2& w);

		Vector2& operator+=(const Vector2& v);
		Vector2& operator-=(const Vector2& v);
		Vector2& operator*=(const Vector2& v);
		Vector2& operator*=(float s);
		Vector2& operator/=(const Vector2& v);
		Vector2& operator/=(float s);
		float operator[](uint64_t index) const;

		float getElement(uint64_t index) const;
		void setElement(uint64_t index, float value);
		float length() const;
		float lengthSquared() const;
		void normalize();
		Vector2 normalized() const;
		void inverse();
		Vector2 inversed() const;
		bool isZero() const;
		bool isNan() const;
		bool isNormal() const;
		float dot(const Vector2& v) const;
		Vector2 reflect(const Vector2& normal) const;
		std::string toString() const;

		static Vector2 lerp(const Vector2& v1, const Vector2& v2, float t);
		static Vector2 abs(const Vector2& v);

		float x;
		float y;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(x),
				CEREAL_NVP(y));
		}
	};
}

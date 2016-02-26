// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIfloat, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Color
	{
	public:

		explicit Color(float r = 0.0f, float g = 0.0f, float b = 0.0f, float a = 1.0f);
		explicit Color(int32_t r, int32_t g, int32_t b, int32_t a = 255);

		friend Color operator+(const Color& c1, const Color& c2);
		friend Color operator-(const Color& c1, const Color& c2);
		friend Color operator*(const Color& c1, const Color& c2);
		friend Color operator*(const Color& c, float s);
		friend Color operator*(float s, const Color& c);
		friend Color operator/(const Color& c1, const Color& c2);
		friend Color operator/(const Color& c, float s);
		friend bool operator==(const Color& c1, const Color& c2);
		friend bool operator!=(const Color& c1, const Color& c2);
		
		Color& operator+=(const Color& c);
		Color& operator-=(const Color& c);
		Color& operator*=(const Color& c);
		Color& operator*=(float s);
		Color& operator/=(float s);

		uint32_t getRgbaValue() const;
		uint32_t getAbgrValue() const;
		float getLuminance() const;
		bool isTransparent() const;
		bool isZero() const;
		bool isClamped() const;
		bool isNan() const;
		bool isNegative() const;
		Color& clamp();
		Color clamped() const;
		
		static Color fromRgbaValue(uint32_t rgba);
		static Color fromAbgrValue(uint32_t abgr);
		static Color lerp(const Color& start, const Color& end, float alpha);
		static Color alphaBlend(const Color& first, const Color& second);
		static Color pow(const Color& color, float power);
		static Color fastPow(const Color& color, float power);

		static const Color RED;
		static const Color GREEN;
		static const Color BLUE;
		static const Color WHITE;
		static const Color BLACK;

		float r;
		float g;
		float b;
		float a;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(r),
				CEREAL_NVP(g),
				CEREAL_NVP(b),
				CEREAL_NVP(a));
		}
	};
}

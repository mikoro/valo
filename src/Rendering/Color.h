// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <random>

#include "cereal/cereal.hpp"

namespace Raycer
{
	template <typename T>
	class ColorType
	{
	public:

		explicit ColorType(T r = T(0.0), T g = T(0.0), T b = T(0.0), T a = T(1.0));
		explicit ColorType(int32_t r, int32_t g, int32_t b, int32_t a = 255);

		ColorType<T>& operator+=(const ColorType<T>& c);
		ColorType<T>& operator-=(const ColorType<T>& c);
		ColorType<T>& operator*=(T s);
		ColorType<T>& operator/=(T s);

		uint32_t getRgbaValue() const;
		uint32_t getAbgrValue() const;
		T getLuminance() const;
		bool isTransparent() const;
		bool isZero() const;
		bool isClamped() const;
		bool isNan() const;
		bool isNegative() const;
		ColorType<T>& clamp();
		ColorType<T> clamped() const;

		static ColorType<T> fromRgbaValue(uint32_t rgba);
		static ColorType<T> fromAbgrValue(uint32_t abgr);
		static ColorType<T> lerp(const ColorType<T>& start, const ColorType<T>& end, T alpha);
		static ColorType<T> alphaBlend(const ColorType<T>& first, const ColorType<T>& second);
		static ColorType<T> pow(const ColorType<T>& color, T power);
		static ColorType<T> fastPow(const ColorType<T>& color, T power);
		static ColorType<T> random(std::mt19937& generator);

		static const ColorType<T> RED;
		static const ColorType<T> GREEN;
		static const ColorType<T> BLUE;
		static const ColorType<T> WHITE;
		static const ColorType<T> BLACK;

		T r;
		T g;
		T b;
		T a;

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

	using Color = ColorType<double>;
	using Colorf = ColorType<float>;
}

#include "Rendering/Color.inl"

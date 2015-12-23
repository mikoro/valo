// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Math/MathUtils.h"

namespace Raycer
{
	template <typename T>
	const ColorType<T> ColorType<T>::RED = ColorType<T>(T(1.0), T(0.0), T(0.0));

	template <typename T>
	const ColorType<T> ColorType<T>::GREEN = ColorType<T>(T(0.0), T(1.0), T(0.0));

	template <typename T>
	const ColorType<T> ColorType<T>::BLUE = ColorType<T>(T(0.0), T(0.0), T(1.0));

	template <typename T>
	const ColorType<T> ColorType<T>::WHITE = ColorType<T>(T(1.0), T(1.0), T(1.0));

	template <typename T>
	const ColorType<T> ColorType<T>::BLACK = ColorType<T>(T(0.0), T(0.0), T(0.0));

	template <typename T>
	ColorType<T>::ColorType(T r_, T g_, T b_, T a_) : r(r_), g(g_), b(b_), a(a_)
	{
	}

	template <typename T>
	ColorType<T>::ColorType(int32_t r_, int32_t g_, int32_t b_, int32_t a_)
	{
		assert(r_ >= 0 && r_ <= 255 && g_ >= 0 && g_ <= 255 && b_ >= 0 && b_ <= 255 && a_ >= 0 && a_ <= 255);

		const T inv255 = T(1.0 / 255.0);

		r = T(r_) * inv255;
		g = T(g_) * inv255;
		b = T(b_) * inv255;
		a = T(a_) * inv255;
	}

	template <typename T>
	ColorType<T> operator+(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return ColorType<T>(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b, c1.a + c2.a);
	};

	template <typename T>
	ColorType<T> operator-(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return ColorType<T>(c1.r - c2.r, c1.g - c2.g, c1.b - c2.b, c1.a - c2.a);
	};

	template <typename T>
	ColorType<T> operator*(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return ColorType<T>(c1.r * c2.r, c1.g * c2.g, c1.b * c2.b, c1.a * c2.a);
	};

	template <typename T>
	ColorType<T> operator*(const ColorType<T>& c, T s)
	{
		return ColorType<T>(c.r * s, c.g * s, c.b * s, c.a * s);
	};

	template <typename T>
	ColorType<T> operator*(T s, const ColorType<T>& c)
	{
		return ColorType<T>(c.r * s, c.g * s, c.b * s, c.a * s);
	};

	template <typename T>
	ColorType<T> operator/(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return ColorType<T>(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b, c1.a / c2.a);
	};

	template <typename T>
	ColorType<T> operator/(const ColorType<T>& c, T s)
	{
		T invS = T(1.0) / s;
		return ColorType<T>(c.r * invS, c.g * invS, c.b * invS, c.a * invS);
	};

	template <typename T>
	bool operator==(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return MathUtils::almostSame(c1.r, c2.r) && MathUtils::almostSame(c1.g, c2.g) && MathUtils::almostSame(c1.b, c2.b) && MathUtils::almostSame(c1.a, c2.a);
	};

	template <typename T>
	bool operator!=(const ColorType<T>& c1, const ColorType<T>& c2)
	{
		return !(c1 == c2);
	};

	template <typename T>
	ColorType<T>& ColorType<T>::operator+=(const ColorType<T>& c)
	{
		*this = *this + c;
		return *this;
	}

	template <typename T>
	ColorType<T>& ColorType<T>::operator-=(const ColorType<T>& c)
	{
		*this = *this - c;
		return *this;
	}

	template <typename T>
	ColorType<T>& ColorType<T>::operator*=(T s)
	{
		*this = *this * s;
		return *this;
	}

	template <typename T>
	ColorType<T>& ColorType<T>::operator/=(T s)
	{
		*this = *this / s;
		return *this;
	}

	template <typename T>
	uint32_t ColorType<T>::getRgbaValue() const
	{
		assert(isClamped());

		uint32_t r_ = static_cast<uint32_t>(r * T(255.0) + T(0.5)) & 0xff;
		uint32_t g_ = static_cast<uint32_t>(g * T(255.0) + T(0.5)) & 0xff;
		uint32_t b_ = static_cast<uint32_t>(b * T(255.0) + T(0.5)) & 0xff;
		uint32_t a_ = static_cast<uint32_t>(a * T(255.0) + T(0.5)) & 0xff;

		return (r_ << 24 | g_ << 16 | b_ << 8 | a_);
	}

	template <typename T>
	uint32_t ColorType<T>::getAbgrValue() const
	{
		assert(isClamped());

		uint32_t r_ = static_cast<uint32_t>(r * T(255.0) + T(0.5)) & 0xff;
		uint32_t g_ = static_cast<uint32_t>(g * T(255.0) + T(0.5)) & 0xff;
		uint32_t b_ = static_cast<uint32_t>(b * T(255.0) + T(0.5)) & 0xff;
		uint32_t a_ = static_cast<uint32_t>(a * T(255.0) + T(0.5)) & 0xff;

		return (a_ << 24 | b_ << 16 | g_ << 8 | r_);
	}

	template <typename T>
	T ColorType<T>::getLuminance() const
	{
		return T(0.2126) * r + T(0.7152) * g + T(0.0722) * b; // expects linear space
	}

	template <typename T>
	bool ColorType<T>::isTransparent() const
	{
		return (a < T(1.0));
	}

	template <typename T>
	bool ColorType<T>::isZero() const
	{
		return (r == T(0.0) && g == T(0.0) && b == T(0.0)); // ignore alpha
	}

	template <typename T>
	bool ColorType<T>::isClamped() const
	{
		return (r >= T(0.0) && r <= T(1.0) && g >= T(0.0) && g <= T(1.0) && b >= T(0.0) && b <= T(1.0) && a >= T(0.0) && a <= T(1.0));
	}

	template <typename T>
	bool ColorType<T>::isNan() const
	{
		return (std::isnan(r) || std::isnan(g) || std::isnan(b) || std::isnan(a));
	}

	template <typename T>
	bool ColorType<T>::isNegative() const
	{
		return (r < T(0.0) || g < T(0.0) || b < T(0.0) || a < T(0.0));
	}

	template <typename T>
	ColorType<T>& ColorType<T>::clamp()
	{
		*this = clamped();
		return *this;
	}

	template <typename T>
	ColorType<T> ColorType<T>::clamped() const
	{
		ColorType<T> c;

		c.r = std::max(T(0.0), std::min(r, T(1.0)));
		c.g = std::max(T(0.0), std::min(g, T(1.0)));
		c.b = std::max(T(0.0), std::min(b, T(1.0)));
		c.a = std::max(T(0.0), std::min(a, T(1.0)));

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::fromRgbaValue(uint32_t rgba)
	{
		const T inv255 = T(1.0 / 255.0);

		uint32_t r_ = (rgba >> 24);
		uint32_t g_ = (rgba >> 16) & 0xff;
		uint32_t b_ = (rgba >> 8) & 0xff;
		uint32_t a_ = rgba & 0xff;

		ColorType<T> c;

		c.r = T(r_) * inv255;
		c.g = T(g_) * inv255;
		c.b = T(b_) * inv255;
		c.a = T(a_) * inv255;

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::fromAbgrValue(uint32_t abgr)
	{
		const T inv255 = T(1.0 / 255.0);

		uint32_t r_ = abgr & 0xff;
		uint32_t g_ = (abgr >> 8) & 0xff;
		uint32_t b_ = (abgr >> 16) & 0xff;
		uint32_t a_ = (abgr >> 24);

		ColorType<T> c;

		c.r = T(r_) * inv255;
		c.g = T(g_) * inv255;
		c.b = T(b_) * inv255;
		c.a = T(a_) * inv255;

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::lerp(const ColorType<T>& start, const ColorType<T>& end, T alpha)
	{
		ColorType<T> c;

		c.r = start.r + (end.r - start.r) * alpha;
		c.g = start.g + (end.g - start.g) * alpha;
		c.b = start.b + (end.b - start.b) * alpha;
		c.a = start.a + (end.a - start.a) * alpha;

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::alphaBlend(const ColorType<T>& first, const ColorType<T>& second)
	{
		const T alpha = second.a;
		const T invAlpha = T(1.0) - alpha;

		ColorType<T> c;

		c.r = (alpha * second.r + invAlpha * first.r);
		c.g = (alpha * second.g + invAlpha * first.g);
		c.b = (alpha * second.b + invAlpha * first.b);
		c.a = T(1.0);

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::pow(const ColorType<T>& color, T power)
	{
		ColorType<T> c;

		c.r = std::pow(color.r, power);
		c.g = std::pow(color.g, power);
		c.b = std::pow(color.b, power);
		c.a = color.a;

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::fastPow(const ColorType<T>& color, T power)
	{
		ColorType<T> c;

		c.r = T(MathUtils::fastPow(double(color.r), double(power)));
		c.g = T(MathUtils::fastPow(double(color.g), double(power)));
		c.b = T(MathUtils::fastPow(double(color.b), double(power)));
		c.a = color.a;

		return c;
	}

	template <typename T>
	ColorType<T> ColorType<T>::random(std::mt19937& generator)
	{
		std::uniform_real_distribution<T> random(T(0.0), T(1.0));

		ColorType<T> c;

		c.r = random(generator);
		c.g = random(generator);
		c.b = random(generator);
		c.a = T(1.0);

		return c;
	}
}

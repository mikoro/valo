// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Rendering/Color.h"
#include "Math/MathUtils.h"

using namespace Raycer;

const Color Color::RED = Color(1.0f, 0.0f, 0.0f);
const Color Color::GREEN = Color(0.0f, 1.0f, 0.0f);
const Color Color::BLUE = Color(0.0f, 0.0f, 1.0f);
const Color Color::WHITE = Color(1.0f, 1.0f, 1.0f);
const Color Color::BLACK = Color(0.0f, 0.0f, 0.0f);

Color::Color(float r_, float g_, float b_, float a_) : r(r_), g(g_), b(b_), a(a_)
{
}

Color::Color(int32_t r_, int32_t g_, int32_t b_, int32_t a_)
{
	assert(r_ >= 0 && r_ <= 255 && g_ >= 0 && g_ <= 255 && b_ >= 0 && b_ <= 255 && a_ >= 0 && a_ <= 255);

	const float inv255 = 1.0f / 255.0f;

	r = r_ * inv255;
	g = g_ * inv255;
	b = b_ * inv255;
	a = a_ * inv255;
}

namespace Raycer
{
	Color operator+(const Color& c1, const Color& c2)
	{
		return Color(c1.r + c2.r, c1.g + c2.g, c1.b + c2.b, c1.a + c2.a);
	}

	Color operator-(const Color& c1, const Color& c2)
	{
		return Color(c1.r - c2.r, c1.g - c2.g, c1.b - c2.b, c1.a - c2.a);
	}

	Color operator*(const Color& c1, const Color& c2)
	{
		return Color(c1.r * c2.r, c1.g * c2.g, c1.b * c2.b, c1.a * c2.a);
	}

	Color operator*(const Color& c, float s)
	{
		return Color(c.r * s, c.g * s, c.b * s, c.a * s);
	}

	Color operator*(float s, const Color& c)
	{
		return Color(c.r * s, c.g * s, c.b * s, c.a * s);
	}

	Color operator/(const Color& c1, const Color& c2)
	{
		return Color(c1.r / c2.r, c1.g / c2.g, c1.b / c2.b, c1.a / c2.a);
	}

	Color operator/(const Color& c, float s)
	{
		float invS = 1.0f / s;
		return Color(c.r * invS, c.g * invS, c.b * invS, c.a * invS);
	}

	bool operator==(const Color& c1, const Color& c2)
	{
		return MathUtils::almostSame(c1.r, c2.r) && MathUtils::almostSame(c1.g, c2.g) && MathUtils::almostSame(c1.b, c2.b) && MathUtils::almostSame(c1.a, c2.a);
	}

	bool operator!=(const Color& c1, const Color& c2)
	{
		return !(c1 == c2);
	}
}

Color& Color::operator+=(const Color& c)
{
	*this = *this + c;
	return *this;
}

Color& Color::operator-=(const Color& c)
{
	*this = *this - c;
	return *this;
}

Color& Color::operator*=(const Color& c)
{
	*this = *this * c;
	return *this;
}

Color& Color::operator*=(float s)
{
	*this = *this * s;
	return *this;
}

Color& Color::operator/=(float s)
{
	*this = *this / s;
	return *this;
}

uint32_t Color::getRgbaValue() const
{
	assert(isClamped());

	uint32_t r_ = static_cast<uint32_t>(r * 255.0f + 0.5f) & 0xff;
	uint32_t g_ = static_cast<uint32_t>(g * 255.0f + 0.5f) & 0xff;
	uint32_t b_ = static_cast<uint32_t>(b * 255.0f + 0.5f) & 0xff;
	uint32_t a_ = static_cast<uint32_t>(a * 255.0f + 0.5f) & 0xff;

	return (r_ << 24 | g_ << 16 | b_ << 8 | a_);
}

uint32_t Color::getAbgrValue() const
{
	assert(isClamped());

	uint32_t r_ = static_cast<uint32_t>(r * 255.0f + 0.5f) & 0xff;
	uint32_t g_ = static_cast<uint32_t>(g * 255.0f + 0.5f) & 0xff;
	uint32_t b_ = static_cast<uint32_t>(b * 255.0f + 0.5f) & 0xff;
	uint32_t a_ = static_cast<uint32_t>(a * 255.0f + 0.5f) & 0xff;

	return (a_ << 24 | b_ << 16 | g_ << 8 | r_);
}

float Color::getLuminance() const
{
	return 0.2126f * r + 0.7152f * g + 0.0722f * b; // expects linear space
}

bool Color::isTransparent() const
{
	return (a < 1.0f);
}

bool Color::isZero() const
{
	return (r == 0.0f && g == 0.0f && b == 0.0f); // ignore alpha
}

bool Color::isClamped() const
{
	return (r >= 0.0f && r <= 1.0f && g >= 0.0f && g <= 1.0f && b >= 0.0f && b <= 1.0f && a >= 0.0f && a <= 1.0f);
}

bool Color::isNan() const
{
	return (std::isnan(r) || std::isnan(g) || std::isnan(b) || std::isnan(a));
}

bool Color::isNegative() const
{
	return (r < 0.0f || g < 0.0f || b < 0.0f || a < 0.0f);
}

Color& Color::clamp()
{
	*this = clamped();
	return *this;
}

Color Color::clamped() const
{
	Color c;

	c.r = std::max(0.0f, std::min(r, 1.0f));
	c.g = std::max(0.0f, std::min(g, 1.0f));
	c.b = std::max(0.0f, std::min(b, 1.0f));
	c.a = std::max(0.0f, std::min(a, 1.0f));

	return c;
}

Color Color::fromRgbaValue(uint32_t rgba)
{
	const float inv255 = 1.0f / 255.0f;

	uint32_t r_ = (rgba >> 24);
	uint32_t g_ = (rgba >> 16) & 0xff;
	uint32_t b_ = (rgba >> 8) & 0xff;
	uint32_t a_ = rgba & 0xff;

	Color c;

	c.r = r_ * inv255;
	c.g = g_ * inv255;
	c.b = b_ * inv255;
	c.a = a_ * inv255;

	return c;
}

Color Color::fromAbgrValue(uint32_t abgr)
{
	const float inv255 = 1.0f / 255.0f;

	uint32_t r_ = abgr & 0xff;
	uint32_t g_ = (abgr >> 8) & 0xff;
	uint32_t b_ = (abgr >> 16) & 0xff;
	uint32_t a_ = (abgr >> 24);

	Color c;

	c.r = r_ * inv255;
	c.g = g_ * inv255;
	c.b = b_ * inv255;
	c.a = a_ * inv255;

	return c;
}

Color Color::lerp(const Color& start, const Color& end, float alpha)
{
	Color c;

	c.r = start.r + (end.r - start.r) * alpha;
	c.g = start.g + (end.g - start.g) * alpha;
	c.b = start.b + (end.b - start.b) * alpha;
	c.a = start.a + (end.a - start.a) * alpha;

	return c;
}

Color Color::alphaBlend(const Color& first, const Color& second)
{
	const float alpha = second.a;
	const float invAlpha = 1.0f - alpha;

	Color c;

	c.r = (alpha * second.r + invAlpha * first.r);
	c.g = (alpha * second.g + invAlpha * first.g);
	c.b = (alpha * second.b + invAlpha * first.b);
	c.a = 1.0f;

	return c;
}

Color Color::pow(const Color& color, float power)
{
	Color c;

	c.r = std::pow(color.r, power);
	c.g = std::pow(color.g, power);
	c.b = std::pow(color.b, power);
	c.a = color.a;

	return c;
}

Color Color::fastPow(const Color& color, float power)
{
	Color c;

	c.r = MathUtils::fastPow(color.r, power);
	c.g = MathUtils::fastPow(color.g, power);
	c.b = MathUtils::fastPow(color.b, power);
	c.a = color.a;

	return c;
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/AabbSimd.h"
#include "Tracing/Aabb.h"

using namespace Raycer;

AabbSimd::AabbSimd()
{
	const float fmin = std::numeric_limits<float>::max();
	const float fmax = std::numeric_limits<float>::lowest();

	min = _mm_set_ps(0.0f, fmin, fmin, fmin);
	max = _mm_set_ps(0.0f, fmax, fmax, fmax);
}

AabbSimd::AabbSimd(const Aabb& aabb)
{
	min = _mm_set_ps(0.0f, aabb.min.z, aabb.min.y, aabb.min.x);
	max = _mm_set_ps(0.0f, aabb.max.z, aabb.max.y, aabb.max.x);
}

void AabbSimd::expand(const AabbSimd& other)
{
	min = _mm_min_ps(min, other.min);
	max = _mm_max_ps(max, other.max);
}

/*float AabbSimd::getSurfaceArea() const
{
	__m128 extent = _mm_sub_ps(max, min); // x y z 0
	__m128 extent1 = _mm_shuffle_ps(extent, extent, _MM_SHUFFLE(3, 0, 2, 0)); // x z x 0
	__m128 extent2 = _mm_shuffle_ps(extent, extent, _MM_SHUFFLE(3, 2, 1, 1)); // y y z 0
	__m128 mul = _mm_mul_ps(extent1, extent2);

	// horizontal add
	__m128 shuf = _mm_movehdup_ps(mul);
	__m128 sums = _mm_add_ps(mul, shuf);
	shuf = _mm_movehl_ps(shuf, sums);
	sums = _mm_add_ss(sums, shuf);

	return 2.0f * _mm_cvtss_f32(sums);
}*/

float AabbSimd::getSurfaceArea() const
{
	alignas(16) float min_[4];
	alignas(16) float max_[4];

	_mm_store_ps(min_, min);
	_mm_store_ps(max_, max);

	float extent[3];

	extent[0] = max_[0] - min_[0];
	extent[1] = max_[1] - min_[1];
	extent[2] = max_[2] - min_[2];

	return 2.0f * (extent[0] * extent[1] + extent[2] * extent[1] + extent[0] * extent[2]);
}

Aabb AabbSimd::getAabb() const
{
	alignas(16) float min_[4];
	alignas(16) float max_[4];

	_mm_store_ps(min_, min);
	_mm_store_ps(max_, max);

	Aabb aabb;

	aabb.min.x = min_[0];
	aabb.min.y = min_[1];
	aabb.min.z = min_[2];
	aabb.max.x = max_[0];
	aabb.max.y = max_[1];
	aabb.max.z = max_[2];

	return aabb;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/Mapper.h"
#include "Math/Vector2.h"

using namespace Raycer;

CUDA_CALLABLE Vector2 Mapper::mapToDisc(const Vector2& point)
{
	float phi, r;
	float a = 2.0f * point.x - 1.0f;
	float b = 2.0f * point.y - 1.0f;

	if (a > -b)
	{
		if (a > b)
		{
			r = a;
			phi = (float(M_PI) / 4.0f) * (b / a);
		}
		else
		{
			r = b;
			phi = (float(M_PI) / 4.0f) * (2.0f - (a / b));
		}
	}
	else
	{
		if (a < b)
		{
			r = -a;
			phi = (float(M_PI) / 4.0f) * (4.0f + (b / a));
		}
		else
		{
			r = -b;

			if (b != 0.0f)
				phi = (float(M_PI) / 4.0f) * (6.0f - (a / b));
			else
				phi = 0.0f;
		}
	}

	float u = r * std::cos(phi);
	float v = r * std::sin(phi);

	return Vector2(u, v);
}

CUDA_CALLABLE Vector3 Mapper::mapToCosineHemisphere(const Vector2& point, const ONB& onb)
{
	Vector2 discPoint = mapToDisc(point);

	float r2 = discPoint.x * discPoint.x + discPoint.y * discPoint.y;

	if (r2 > 1.0f)
		r2 = 1.0f;

	float x = discPoint.x;
	float y = discPoint.y;
	float z = std::sqrt(1.0f - r2);

	return x * onb.u + y * onb.v + z * onb.w;
}

CUDA_CALLABLE Vector3 Mapper::mapToUniformHemisphere(const Vector2& point, const ONB& onb)
{
	Vector2 discPoint = mapToDisc(point);

	float r2 = discPoint.x * discPoint.x + discPoint.y * discPoint.y;
	float a = std::sqrt(2.0f - r2);

	float x = discPoint.x * a;
	float y = discPoint.y * a;
	float z = 1.0f - r2;

	return x * onb.u + y * onb.v + z * onb.w;
}

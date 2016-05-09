// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/BellFilter.h"
#include "Math/Vector2.h"

using namespace Valo;

namespace
{
	CUDA_CALLABLE float calculateWeight(float s)
	{
		s = std::abs(s);

		if (s < 0.5f)
			return 0.75f - (s * s);
		
		if (s <= 1.5f)
		{
			s = s - 1.5f;
			return 0.5f * (s * s);
		}
		
		return 0.0f;
	}
}

CUDA_CALLABLE float BellFilter::getWeight(float s) const
{
	return calculateWeight(s);
}

CUDA_CALLABLE float BellFilter::getWeight(const Vector2& point) const
{
	return calculateWeight(point.x) * calculateWeight(point.y);
}

CUDA_CALLABLE Vector2 BellFilter::getRadius() const
{
	return Vector2(1.5f, 1.5f);
}

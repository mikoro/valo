// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class BellFilter : public Filter
	{
	public:

		BellFilter();

		float getWeightX(float x) override;
		float getWeightY(float y) override;
	};
}

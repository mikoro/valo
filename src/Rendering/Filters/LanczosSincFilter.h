// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Rendering/Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class LanczosSincFilter : public Filter
	{
	public:

		LanczosSincFilter(double width = 2.0);

		double getWeight(double x);
		double getWeight(double x, double y);
		double getWeight(const Vector2& point);
		double getWidth();
	};
}
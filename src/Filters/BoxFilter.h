// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class BoxFilter : public Filter
	{
	public:

		explicit BoxFilter(float radiusX = 0.5f, float radiusY = 0.5f);

		void setRadius(float radiusX, float radiusY);

		float getWeightX(float x) override;
		float getWeightY(float y) override;

	private:

		float weightX = 0.0f;
		float weightY = 0.0f;
	};
}

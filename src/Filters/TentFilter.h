// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class TentFilter : public Filter
	{
	public:

		explicit TentFilter(float radiusX = 1.0f, float radiusY = 1.0f);

		void setRadius(float radiusX, float radiusY);

		float getWeightX(float x) override;
		float getWeightY(float y) override;

	private:

		float radiusXInv = 0.0f;
		float radiusYInv = 0.0f;
	};
}

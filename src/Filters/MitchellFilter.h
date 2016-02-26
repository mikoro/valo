// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class MitchellFilter : public Filter
	{
	public:

		explicit MitchellFilter(float B = (1.0f / 3.0f), float C = (1.0f / 3.0f));

		void setCoefficients(float B, float C);

		float getWeightX(float x) override;
		float getWeightY(float y) override;

	private:

		float B = 0.0f;
		float C = 0.0f;
	};
}

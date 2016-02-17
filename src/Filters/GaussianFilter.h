// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Filters/Filter.h"

namespace Raycer
{
	class Vector2;

	class GaussianFilter : public Filter
	{
	public:

		explicit GaussianFilter(float stdDevX = 1.0f, float stdDevY = 1.0f);

		void setStandardDeviations(float stdDevX, float stdDevY);

		float getWeightX(float x) override;
		float getWeightY(float y) override;

	private:

		float alphaX = 0.0f;
		float alphaY = 0.0f;
		float betaX = 0.0f;
		float betaY = 0.0f;
	};
}

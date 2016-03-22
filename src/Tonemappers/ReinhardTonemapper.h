// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Math/MovingAverage.h"

// https://www.cs.utah.edu/~reinhard/cdrom/tonemap.pdf

namespace Raycer
{
	class Image;

	class ReinhardTonemapper
	{
	public:

		ReinhardTonemapper();

		void apply(const Image& inputImage, Image& outputImage);

		bool applyGamma = true;
		bool shouldClamp = true;
		float gamma = 2.2f;
		float key = 0.18f;
		bool enableAveraging = true;
		float averagingAlpha = 0.001f;

	private:

		MovingAverage maxLuminanceAverage;
	};
}

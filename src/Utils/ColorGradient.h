// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "Math/Color.h"
#include "Core/Common.h"
#include "Utils/CudaAlloc.h"

namespace Valo
{
	struct ColorGradientSegment
	{
		Color startColor;
		Color endColor;
		uint32_t startIndex = 0;
		uint32_t endIndex = 0;
	};

	class ColorGradient
	{
	public:

		ColorGradient();

		void addSegment(const Color& start, const Color& end, uint32_t length);
		void commit();
		CUDA_CALLABLE Color getColor(float alpha) const;

	private:

		uint32_t segmentCount = 0;
		uint32_t totalLength = 0;
		std::vector<ColorGradientSegment> segments;
		CudaAlloc<ColorGradientSegment> segmentsAlloc;
	};
}

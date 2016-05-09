// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/ColorGradient.h"

using namespace Valo;

ColorGradient::ColorGradient() : segmentsAlloc(false)
{
}

void ColorGradient::addSegment(const Color& start, const Color& end, uint32_t length)
{
	assert(length >= 1);

	ColorGradientSegment segment;

	segment.startColor = start;
	segment.endColor = end;
	segment.startIndex = totalLength;
	segment.endIndex = totalLength + length;

	segments.push_back(segment);

	segmentCount = uint32_t(segments.size());
	totalLength += length;
}

void ColorGradient::commit()
{
	segmentsAlloc.resize(segments.size());
	segmentsAlloc.write(segments.data(), segments.size());
}

CUDA_CALLABLE Color ColorGradient::getColor(float alpha) const
{
	assert(alpha >= 0.0f && alpha <= 1.0f);

	Color result;
	uint32_t index = uint32_t(ceil(alpha * totalLength));
	ColorGradientSegment* segmentsPtr = segmentsAlloc.getPtr();

	for (uint32_t i = 0; i < segmentCount; ++i)
	{
		ColorGradientSegment& segment = segmentsPtr[i];

		if (index >= segment.startIndex && index <= segment.endIndex)
		{
			float alphaStart = segment.startIndex / float(totalLength);
			float alphaEnd = segment.endIndex / float(totalLength);
			float segmentAlpha = (alpha - alphaStart) / (alphaEnd - alphaStart);

			result = Color::lerp(segment.startColor, segment.endColor, segmentAlpha);

			break;
		}
	}

	return result;
}

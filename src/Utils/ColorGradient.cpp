// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/ColorGradient.h"

using namespace Raycer;

void ColorGradient::addSegment(const Color& start, const Color& end, uint64_t length)
{
	assert(length >= 1);

	ColorGradientSegment segment;

	segment.startColor = start;
	segment.endColor = end;
	segment.startIndex = totalLength;
	segment.endIndex = totalLength + length;

	segments.push_back(segment);

	totalLength += length;
}

Color ColorGradient::getColor(float alpha) const
{
	assert(alpha >= 0.0f && alpha <= 1.0f);

	Color result;
	uint64_t index = uint64_t(std::ceil(alpha * totalLength));

	for (const ColorGradientSegment& segment : segments)
	{
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

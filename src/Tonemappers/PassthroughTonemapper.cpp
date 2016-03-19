// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Image.h"
#include "Math/Color.h"
#include "Tonemappers/PassthroughTonemapper.h"

using namespace Raycer;

void PassthroughTonemapper::apply(const Image& inputImage, Image& outputImage)
{
	const Color* inputPixels = inputImage.getPixelData();
	Color* outputPixels = outputImage.getPixelData();
	int32_t pixelCount = inputImage.getLength();

	for (int32_t i = 0; i < pixelCount; ++i)
		outputPixels[i] = inputPixels[i];
}

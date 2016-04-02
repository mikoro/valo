// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Image.h"
#include "Math/Color.h"
#include "Math/MathUtils.h"
#include "Tonemappers/LinearTonemapper.h"

using namespace Raycer;

void LinearTonemapper::apply(const Image& inputImage, Image& outputImage)
{
	const Color* inputPixels = inputImage.getData();
	Color* outputPixels = outputImage.getData();
	int32_t pixelCount = inputImage.getLength();

	float invGamma = 1.0f / gamma;

	#pragma omp parallel for
	for (int32_t i = 0; i < pixelCount; ++i)
	{
		Color outputColor = inputPixels[i];
		outputColor *= MathUtils::fastPow(2.0f, exposure);

		if (shouldClamp)
			outputColor.clamp();

		if (applyGamma)
			outputColor = Color::fastPow(outputColor, invGamma);

		outputColor.a = 1.0f;
		outputPixels[i] = outputColor;
	}
}

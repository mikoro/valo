// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Image.h"
#include "Math/Color.h"
#include "Tonemappers/ReinhardTonemapper.h"

using namespace Raycer;

ReinhardTonemapper::ReinhardTonemapper()
{
	maxLuminanceAverage.setAverage(1.0f);
}

void ReinhardTonemapper::apply(const Image& inputImage, Image& outputImage)
{
	const Color* inputPixels = inputImage.getData();
	Color* outputPixels = outputImage.getData();
	int32_t pixelCount = inputImage.getLength();

	float epsilon = 0.01f;
	float luminanceLogSum = 0.0f;
	float maxLuminance = 1.0f;
	float maxLuminancePrivate = 0.0f;
	(void)maxLuminancePrivate; // vs2015 compilation warning fix

	#pragma omp parallel reduction(+:luminanceLogSum) private(maxLuminancePrivate)
	{
		maxLuminancePrivate = 0.0f;

		#pragma omp for
		for (int32_t i = 0; i < pixelCount; ++i)
		{
			float luminance = inputPixels[i].getLuminance();
			luminanceLogSum += std::log(epsilon + luminance);

			if (luminance > maxLuminancePrivate)
				maxLuminancePrivate = luminance;
		}

		if (maxLuminancePrivate > maxLuminance)
		{
			#pragma omp critical
			{
				if (maxLuminancePrivate > maxLuminance)
					maxLuminance = maxLuminancePrivate;
			}
		}
	}

	if (enableAveraging)
	{
		maxLuminanceAverage.setAlpha(averagingAlpha);
		maxLuminanceAverage.addMeasurement(maxLuminance);
		maxLuminance = maxLuminanceAverage.getAverage();
	}

	const float luminanceLogAvg = std::exp(luminanceLogSum / float(pixelCount));
	const float luminanceScale = key / luminanceLogAvg;
	const float maxLuminance2Inv = 1.0f / (maxLuminance * maxLuminance);
	const float invGamma = 1.0f / gamma;

	#pragma omp parallel for
	for (int32_t i = 0; i < pixelCount; ++i)
	{
		Color inputColor = inputPixels[i];

		float originalLuminance = inputColor.getLuminance();
		float scaledLuminance = luminanceScale * originalLuminance;
		float mappedLuminance = (scaledLuminance * (1.0f + (scaledLuminance * maxLuminance2Inv))) / (1.0f + scaledLuminance);
		float colorScale = mappedLuminance / originalLuminance;

		Color outputColor = inputColor * colorScale;

		if (shouldClamp)
			outputColor.clamp();

		if (applyGamma)
			outputColor = Color::fastPow(outputColor, invGamma);

		outputColor.a = 1.0f;
		outputPixels[i] = outputColor;
	}
}

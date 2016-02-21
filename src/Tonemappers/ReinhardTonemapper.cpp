// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tonemappers/ReinhardTonemapper.h"
#include "Tracing/Scene.h"
#include "Rendering/Image.h"
#include "Rendering/Color.h"
#include "App.h"
#include "Utils/Settings.h"

using namespace Raycer;

ReinhardTonemapper::ReinhardTonemapper()
{
	maxLuminanceAverage.setAverage(1.0f);
}

void ReinhardTonemapper::apply(const Scene& scene, const Image& inputImage, Image& outputImage)
{
	Settings& settings = App::getSettings();

	auto& inputPixelData = inputImage.getPixelDataConst();
	auto& outputPixelData = outputImage.getPixelData();

	const float epsilon = 0.01f;
	const int64_t pixelCount = int64_t(inputPixelData.size());
	float luminanceLogSum = 0.0f;
	float maxLuminance = 1.0f;
	float maxLuminancePrivate = 0.0f;
	(void)maxLuminancePrivate; // vs2015 compilation warning fix

	#pragma omp parallel reduction(+:luminanceLogSum) private(maxLuminancePrivate)
	{
		maxLuminancePrivate = 0.0f;

		#pragma omp for
		for (int64_t i = 0; i < pixelCount; ++i)
		{
			float luminance = inputPixelData.at(i).getLuminance();
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

	if (scene.tonemapping.enableAveraging && settings.interactive.enabled)
	{
		maxLuminanceAverage.setAlpha(scene.tonemapping.averagingAlpha);
		maxLuminanceAverage.addMeasurement(maxLuminance);
		maxLuminance = maxLuminanceAverage.getAverage();
	}

	const float luminanceLogAvg = std::exp(luminanceLogSum / float(pixelCount));
	const float luminanceScale = scene.tonemapping.key / luminanceLogAvg;
	const float maxLuminance2Inv = 1.0f / (maxLuminance * maxLuminance);
	const float invGamma = 1.0f / scene.tonemapping.gamma;
	
	#pragma omp parallel for
	for (int64_t i = 0; i < pixelCount; ++i)
	{
		Color inputColor = inputPixelData.at(i);

		float originalLuminance = inputColor.getLuminance();
		float scaledLuminance = luminanceScale * originalLuminance;
		float mappedLuminance = (scaledLuminance * (1.0f + (scaledLuminance * maxLuminance2Inv))) / (1.0f + scaledLuminance);
		float colorScale = mappedLuminance / originalLuminance;

		Color outputColor = inputColor * colorScale;

		if (scene.tonemapping.shouldClamp)
			outputColor.clamp();

		if (scene.tonemapping.applyGamma)
			outputColor = Color::fastPow(outputColor, invGamma);

		outputColor.a = 1.0f;
		outputPixelData[i] = outputColor;
	}
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stdafx.h"

#include "Tonemappers/SimpleTonemapper.h"
#include "Tracing/Scene.h"
#include "Rendering/Image.h"
#include "Rendering/Color.h"
#include "Math/MathUtils.h"

using namespace Raycer;

void SimpleTonemapper::apply(const Scene& scene, const Image& inputImage, Image& outputImage)
{
	const AlignedColorfVector& inputPixelData = inputImage.getPixelDataConst();
	AlignedColorfVector& outputPixelData = outputImage.getPixelData();

	const double invGamma = 1.0 / scene.tonemapper.gamma;

	#pragma omp parallel for
	for (int64_t i = 0; i < int64_t(inputPixelData.size()); ++i)
	{
		Color outputColor = inputPixelData.at(i).toColor();
		outputColor *= MathUtils::fastPow(2.0, scene.tonemapper.exposure);

		outputColor = outputColor / (Color(1.0, 1.0, 1.0, 1.0) + outputColor);

		if (scene.tonemapper.shouldClamp)
			outputColor.clamp();

		if (scene.tonemapper.applyGamma)
			outputColor = Color::fastPow(outputColor, invGamma);

		outputColor.a = 1.0;
		outputPixelData[i] = outputColor.toColorf();
	}
}

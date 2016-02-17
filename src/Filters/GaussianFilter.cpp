// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/GaussianFilter.h"

using namespace Raycer;

GaussianFilter::GaussianFilter(float stdDevX, float stdDevY)
{
	setStandardDeviations(stdDevX, stdDevY);
}

void GaussianFilter::setStandardDeviations(float stdDevX, float stdDevY)
{
	alphaX = 1.0f / (std::sqrt(2.0f * float(M_PI)) * stdDevX);
	alphaY = 1.0f / (std::sqrt(2.0f * float(M_PI)) * stdDevY);
	betaX = -1.0f / (2.0f * stdDevX * stdDevX);
	betaY = -1.0f / (2.0f * stdDevY * stdDevY);
	radiusX = (7.43384f * stdDevX) / 2.0f; // from full width at thousandth of maximum
	radiusY = (7.43384f * stdDevY) / 2.0f;
}

float GaussianFilter::getWeightX(float x)
{
	return alphaX * std::exp(x * x * betaX);
}

float GaussianFilter::getWeightY(float y)
{
	return alphaY * std::exp(y * y * betaY);
}

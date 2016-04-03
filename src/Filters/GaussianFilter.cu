// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/GaussianFilter.h"

using namespace Raycer;

namespace
{
	CUDA_CALLABLE float calculateWeight(float s, float alpha, float beta)
	{
		return alpha * std::exp(s * s * beta);
	}
}

CUDA_CALLABLE float GaussianFilter::getWeight(float s) const
{
	float alpha = 1.0f / (std::sqrt(2.0f * float(M_PI)) * stdDeviation.x);
	float beta = -1.0f / (2.0f * stdDeviation.x * stdDeviation.x);

	return calculateWeight(s, alpha, beta);
}

CUDA_CALLABLE float GaussianFilter::getWeight(const Vector2& point) const
{
	float alphaX = 1.0f / (std::sqrt(2.0f * float(M_PI)) * stdDeviation.x);
	float alphaY = 1.0f / (std::sqrt(2.0f * float(M_PI)) * stdDeviation.y);
	float betaX = -1.0f / (2.0f * stdDeviation.x * stdDeviation.x);
	float betaY = -1.0f / (2.0f * stdDeviation.y * stdDeviation.y);

	return calculateWeight(point.x, alphaX, betaX) * calculateWeight(point.y, alphaY, betaY);
}

CUDA_CALLABLE Vector2 GaussianFilter::getRadius() const
{
	float radiusX = (7.43384f * stdDeviation.x) / 2.0f; // from full width at thousandth of maximum
	float radiusY = (7.43384f * stdDeviation.y) / 2.0f;

	return Vector2(radiusX, radiusY);
}

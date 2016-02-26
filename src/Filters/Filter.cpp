// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/Filter.h"
#include "Filters/BoxFilter.h"
#include "Filters/TentFilter.h"
#include "Filters/BellFilter.h"
#include "Filters/GaussianFilter.h"
#include "Filters/MitchellFilter.h"
#include "Filters/LanczosSincFilter.h"
#include "Math/Vector2.h"

using namespace Raycer;

std::unique_ptr<Filter> Filter::getFilter(FilterType type)
{
	switch (type)
	{
		case FilterType::BOX: return std::make_unique<BoxFilter>();
		case FilterType::TENT: return std::make_unique<TentFilter>();
		case FilterType::BELL: return std::make_unique<BellFilter>();
		case FilterType::GAUSSIAN: return std::make_unique<GaussianFilter>();
		case FilterType::MITCHELL: return std::make_unique<MitchellFilter>();
		case FilterType::LANCZOS_SINC: return std::make_unique<LanczosSincFilter>();
		default: throw std::runtime_error("Unknown filter type");
	}
}

float Filter::getWeight(float x, float y)
{
	return getWeightX(x) * getWeightY(y);
}

float Filter::getWeight(const Vector2& point)
{
	return getWeightX(point.x) * getWeightY(point.y);
}

float Filter::getRadiusX() const
{
	return radiusX;
}

float Filter::getRadiusY() const
{
	return radiusY;
}

Vector2 Filter::getRadius() const
{
	return Vector2(radiusX, radiusY);
}

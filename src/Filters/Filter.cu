// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Filters/Filter.h"
#include "Math/Vector2.h"

using namespace Raycer;

Filter::Filter(FilterType type_)
{
	type = type_;
}

CUDA_CALLABLE float Filter::getWeight(float s) const
{
	switch (type)
	{
		case FilterType::BOX: return boxFilter.getWeight(s);
		case FilterType::TENT: return tentFilter.getWeight(s);
		case FilterType::BELL: return bellFilter.getWeight(s);
		case FilterType::GAUSSIAN: return gaussianFilter.getWeight(s);
		case FilterType::MITCHELL: return mitchellFilter.getWeight(s);
		case FilterType::LANCZOS_SINC: return lanczosSincFilter.getWeight(s);
		default: return 0.0f;
	}
}

CUDA_CALLABLE float Filter::getWeight(const Vector2& point) const
{
	switch (type)
	{
		case FilterType::BOX: return boxFilter.getWeight(point);
		case FilterType::TENT: return tentFilter.getWeight(point);
		case FilterType::BELL: return bellFilter.getWeight(point);
		case FilterType::GAUSSIAN: return gaussianFilter.getWeight(point);
		case FilterType::MITCHELL: return mitchellFilter.getWeight(point);
		case FilterType::LANCZOS_SINC: return lanczosSincFilter.getWeight(point);
		default: return 0.0f;
	}
}

CUDA_CALLABLE Vector2 Filter::getRadius() const
{
	switch (type)
	{
		case FilterType::BOX: return boxFilter.getRadius();
		case FilterType::TENT: return tentFilter.getRadius();
		case FilterType::BELL: return bellFilter.getRadius();
		case FilterType::GAUSSIAN: return gaussianFilter.getRadius();
		case FilterType::MITCHELL: return mitchellFilter.getRadius();
		case FilterType::LANCZOS_SINC: return lanczosSincFilter.getRadius();
		default: return Vector2(0.0f, 0.0f);
	}
}

std::string Filter::getName() const
{
	switch (type)
	{
		case FilterType::BOX: return "box";
		case FilterType::TENT: return "tent";
		case FilterType::BELL: return "bell";
		case FilterType::GAUSSIAN: return "gaussian";
		case FilterType::MITCHELL: return "mitchell";
		case FilterType::LANCZOS_SINC: return "lanczos_sinc";
		default: return "unknown";
	}
}

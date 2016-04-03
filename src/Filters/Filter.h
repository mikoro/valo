// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <string>

#include "Core/Common.h"
#include "Filters/BoxFilter.h"
#include "Filters/TentFilter.h"
#include "Filters/BellFilter.h"
#include "Filters/GaussianFilter.h"
#include "Filters/MitchellFilter.h"
#include "Filters/LanczosSincFilter.h"

namespace Raycer
{
	class Vector2;

	enum class FilterType { BOX, TENT, BELL, GAUSSIAN, MITCHELL, LANCZOS_SINC };

	class Filter
	{
	public:

		explicit Filter(FilterType type = FilterType::MITCHELL);

		CUDA_CALLABLE float getWeight(float s) const;
		CUDA_CALLABLE float getWeight(const Vector2& point) const;

		CUDA_CALLABLE Vector2 getRadius() const;

		std::string getName() const;

		FilterType type = FilterType::MITCHELL;

		BoxFilter boxFilter;
		TentFilter tentFilter;
		BellFilter bellFilter;
		GaussianFilter gaussianFilter;
		MitchellFilter mitchellFilter;
		LanczosSincFilter lanczosSincFilter;
	};
}

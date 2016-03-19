// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

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

		explicit Filter(FilterType type = FilterType::BOX);

		float getWeight(float s);
		float getWeight(const Vector2& point);

		Vector2 getRadius();

		std::string getName() const;

		FilterType type = FilterType::BOX;

		BoxFilter boxFilter;
		TentFilter tentFilter;
		BellFilter bellFilter;
		GaussianFilter gaussianFilter;
		MitchellFilter mitchellFilter;
		LanczosSincFilter lanczosSincFilter;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(type),
				CEREAL_NVP(boxFilter),
				CEREAL_NVP(tentFilter),
				CEREAL_NVP(bellFilter),
				CEREAL_NVP(gaussianFilter),
				CEREAL_NVP(mitchellFilter),
				CEREAL_NVP(lanczosSincFilter));
		}
	};
}

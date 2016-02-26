// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class MovingAverage
	{
	public:

		explicit MovingAverage(float alpha = 1.0f, float average = 0.0f);

		void setAlpha(float alpha);
		void setAverage(float average);
		void addMeasurement(float value);
		float getAverage() const;

	private:

		float alpha;
		float average;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(alpha),
				CEREAL_NVP(average));
		}
	};
}

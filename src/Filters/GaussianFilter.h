// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector2.h"

namespace Raycer
{
	class Vector2;

	class GaussianFilter
	{
	public:

		float getWeight(float s);
		float getWeight(const Vector2& point);

		Vector2 getRadius();

		Vector2 stdDeviation = Vector2(0.5f, 0.5f);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(stdDeviation));
		}
	};
}

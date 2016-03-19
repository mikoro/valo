// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector2.h"

namespace Raycer
{
	class Vector2;

	class MitchellFilter
	{
	public:

		float getWeight(float s);
		float getWeight(const Vector2& point);

		Vector2 getRadius();

		float B = (1.0f / 3.0f);
		float C = (1.0f / 3.0f);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(B),
				CEREAL_NVP(C));
		}
	};
}

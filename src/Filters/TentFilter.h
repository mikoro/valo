// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector2.h"

namespace Raycer
{
	class Vector2;

	class TentFilter
	{
	public:

		float getWeight(float s);
		float getWeight(const Vector2& point);

		Vector2 getRadius();

		Vector2 radius = Vector2(1.0f, 1.0f);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(radius));
		}
	};
}

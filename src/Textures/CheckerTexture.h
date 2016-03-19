// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Color.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class CheckerTexture
	{
	public:

		void initialize();

		Color getColor(const Vector2& texcoord, const Vector3& position);
		
		Color color1;
		Color color2;
		bool stripeMode = false;
		float stripeWidth = 0.05f;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(color1),
				CEREAL_NVP(color2),
				CEREAL_NVP(stripeMode),
				CEREAL_NVP(stripeWidth));
		}
	};
}

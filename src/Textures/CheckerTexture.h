// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Textures/Texture.h"
#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Vector2;
	class Vector3;

	class CheckerTexture : public Texture
	{
	public:

		void initialize(Scene& scene) override;

		Color getColor(const Vector2& texcoord, const Vector3& position) const override;
		double getValue(const Vector2& texcoord, const Vector3& position) const override;
		
		Color color1;
		Color color2;
		bool stripeMode = false;
		double stripeWidth = 0.05;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(cereal::make_nvp("texture", cereal::base_class<Texture>(this)),
				CEREAL_NVP(color1),
				CEREAL_NVP(color2),
				CEREAL_NVP(stripeMode),
				CEREAL_NVP(stripeWidth));
		}
	};
}

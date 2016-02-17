// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Textures/Texture.h"
#include "Utils/PerlinNoise.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Vector2;
	class Vector3;

	class PerlinNoiseTexture : public Texture
	{
	public:

		void initialize(Scene& scene) override;

		Color getColor(const Vector2& texcoord, const Vector3& position) const override;
		float getValue(const Vector2& texcoord, const Vector3& position) const override;
		
		uint64_t seed = 1;
		Vector3 scale = Vector3(10.0f, 10.0f, 10.0f);
		Color baseColor = Color(1.0f, 1.0f, 1.0f);
		bool isFbm = false;
		uint64_t octaves = 4;
		float lacunarity = 2.0f;
		float persistence = 0.5f;

	private:

		PerlinNoise perlinNoise;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(cereal::make_nvp("texture", cereal::base_class<Texture>(this)),
				CEREAL_NVP(seed),
				CEREAL_NVP(scale),
				CEREAL_NVP(baseColor),
				CEREAL_NVP(isFbm),
				CEREAL_NVP(octaves),
				CEREAL_NVP(lacunarity),
				CEREAL_NVP(persistence));
		}
	};
}

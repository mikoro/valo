// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Color.h"
#include "Textures/CheckerTexture.h"
#include "Textures/ImageTexture.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	enum class TextureType { CHECKER, IMAGE };

	class Texture
	{
	public:

		void initialize();

		Color getColor(const Vector2& texcoord, const Vector3& position);
		
		uint32_t id = 0;
		TextureType type = TextureType::CHECKER;

		CheckerTexture checkerTexture;
		ImageTexture imageTexture;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(type),
				CEREAL_NVP(checkerTexture),
				CEREAL_NVP(imageTexture));
		}
	};
}

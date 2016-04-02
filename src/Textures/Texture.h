// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
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

		void initialize(Scene& scene);

		CUDA_CALLABLE Color getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position);
		
		uint32_t id = 0;
		TextureType type = TextureType::CHECKER;

		CheckerTexture checkerTexture;
		ImageTexture imageTexture;
	};
}

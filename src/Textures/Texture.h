// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Math/Color.h"
#include "Textures/ImageTexture.h"
#include "Textures/CheckerTexture.h"
#include "Textures/MarbleTexture.h"
#include "Textures/WoodTexture.h"
#include "Textures/FireTexture.h"

namespace Valo
{
	class Scene;
	class Vector2;
	class Vector3;

	enum class TextureType { IMAGE, CHECKER, MARBLE, WOOD, FIRE };

	class Texture
	{
	public:

		explicit Texture(TextureType type = TextureType::CHECKER);

		void initialize(Scene& scene);

		CUDA_CALLABLE Color getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
		
		uint32_t id = 0;
		TextureType type = TextureType::CHECKER;

		ImageTexture imageTexture;
		CheckerTexture checkerTexture;
		MarbleTexture marbleTexture;
		WoodTexture woodTexture;
		FireTexture fireTexture;
	};
}

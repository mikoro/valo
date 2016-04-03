// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Core/Image.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class ImageTexture
	{
	public:

		void initialize(Scene& scene);

		CUDA_CALLABLE Color getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
		
		std::string imageFileName;
		bool applyGamma = false;

	private:

		uint32_t imageIndex = 0;
	};
}

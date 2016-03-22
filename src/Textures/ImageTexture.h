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

		void initialize();

		CUDA_CALLABLE Color getColor(const Vector2& texcoord, const Vector3& position);
		
		std::string imageFileName;
		bool applyGamma = false;

	private:

		Image* image = nullptr;
	};
}

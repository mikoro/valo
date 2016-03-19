// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Core/Image.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class ImageTexture
	{
	public:

		void initialize();

		Color getColor(const Vector2& texcoord, const Vector3& position);
		
		std::string imageFileName;
		bool applyGamma = false;

	private:

		Image* image = nullptr;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(imageFileName),
				CEREAL_NVP(applyGamma));
		}
	};
}

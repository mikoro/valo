// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Textures/Texture.h"
#include "Rendering/Image.h"

namespace Raycer
{
	class Scene;
	class Vector2;
	class Vector3;

	class ImageTexture : public Texture
	{
	public:

		void initialize(Scene& scene) override;

		Color getColor(const Vector2& texcoord, const Vector3& position) const override;
		double getValue(const Vector2& texcoord, const Vector3& position) const override;
		
		std::string imageFilePath;
		bool applyGamma = false;

	private:

		const Imagef* image = nullptr;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(cereal::make_nvp("texture", cereal::base_class<Texture>(this)),
				CEREAL_NVP(imageFilePath),
				CEREAL_NVP(applyGamma));
		}
	};
}

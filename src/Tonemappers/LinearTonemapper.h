// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Image;

	class LinearTonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);

		bool applyGamma = true;
		bool shouldClamp = true;
		float gamma = 2.2f;
		float exposure = 0.0f;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(applyGamma),
				CEREAL_NVP(shouldClamp),
				CEREAL_NVP(gamma),
				CEREAL_NVP(exposure));
		}
	};
}

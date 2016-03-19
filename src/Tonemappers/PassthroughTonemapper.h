// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Image;

	class PassthroughTonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
		}
	};
}

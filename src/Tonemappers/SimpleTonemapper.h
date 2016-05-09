// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Valo
{
	class Image;

	class SimpleTonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);

		bool applyGamma = true;
		bool shouldClamp = true;
		float gamma = 2.2f;
		float exposure = 0.0f;
	};
}

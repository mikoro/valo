// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Image;

	class PassthroughTonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);
	};
}

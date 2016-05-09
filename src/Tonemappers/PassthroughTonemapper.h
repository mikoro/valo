// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Valo
{
	class Image;

	class PassthroughTonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);
	};
}

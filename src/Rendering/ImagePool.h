// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "Rendering/Image.h"

/*

ImagePool is used by ImageTextures to prevent loading the same file twice to the memory.

*/

namespace Raycer
{
	class ImagePool
	{
	public:

		static const Imagef* loadImage(const std::string& fileName, bool applyGamma);
		static uint64_t getImageIndex(const std::string& fileName);
		static const std::vector<Imagef>& getImages();
		static void clear();

	private:

		static std::map<std::string, uint64_t> imageIndexMap;
		static std::vector<Imagef> images;
		static bool initialized;

		static const uint64_t MAX_IMAGES = 1000;
	};
}

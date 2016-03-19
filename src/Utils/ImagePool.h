// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>
#include <vector>

#include "Core/Image.h"

namespace Raycer
{
	class ImagePool
	{
	public:

		Image* loadImage(const std::string& fileName, bool applyGamma);
		void clear();

	private:

		bool initialized = false;
		std::map<std::string, uint64_t> imageIndexMap;
		std::vector<Image> images;
	};
}

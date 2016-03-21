// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <map>

namespace Raycer
{
	class Image;

	class ImagePool
	{
	public:

		~ImagePool();

		Image* load(const std::string& fileName, bool applyGamma);
		void clear();

	private:

		std::map<std::string, Image*> images;
	};
}

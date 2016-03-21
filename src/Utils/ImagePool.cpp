// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Utils/ImagePool.h"
#include "Core/Image.h"

using namespace Raycer;

ImagePool::~ImagePool()
{
	clear();
}

Image* ImagePool::load(const std::string& fileName, bool applyGamma)
{
	if (!images.count(fileName))
	{
		Image* image = new Image(fileName);
		images[fileName] = image;

		if (applyGamma)
			image->applyFastGamma(2.2f);
	}

	return images[fileName];
}

void ImagePool::clear()
{
	for (auto& kv : images)
		delete kv.second;

	images.clear();
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Utils/ImagePool.h"
#include "Core/Image.h"

using namespace Raycer;

namespace
{
	const uint32_t MAX_IMAGES = 1000;
}

Image* ImagePool::loadImage(const std::string& fileName, bool applyGamma)
{
	if (!initialized)
	{
		images.reserve(MAX_IMAGES);
		initialized = true;
	}

	if (!imageIndexMap.count(fileName))
	{
		images.push_back(Image(fileName));
		imageIndexMap[fileName] = uint32_t(images.size() - 1);

		if (applyGamma)
			images.back().applyFastGamma(2.2f);
	}

	// the limit is arbitrary, increase it if it becomes a problem
	// idea is to prevent push_back from invalidating pointers
	if (images.size() > MAX_IMAGES)
		throw std::runtime_error("Image pool maximum size exceeded");

	return &images[imageIndexMap[fileName]];
}

void ImagePool::clear()
{
	imageIndexMap.clear();
	images.clear();
}

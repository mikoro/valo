// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Rendering/ImagePool.h"
#include "Rendering/Image.h"

using namespace Raycer;

namespace
{
	const uint64_t MAX_IMAGES = 1000;
}

const Image* ImagePool::getImage(const std::string& fileName, bool applyGamma)
{
	if (!initialized)
	{
		images.reserve(MAX_IMAGES);
		initialized = true;
	}

	if (!imageIndexMap.count(fileName))
	{
		images.push_back(Image(fileName));
		imageIndexMap[fileName] = images.size() - 1;

		if (applyGamma)
			images.back().applyFastGamma(2.2f);
	}

	// the limit is arbitrary, increase it if it becomes a problem
	// idea is to prevent push_back from invalidating pointers
	if (images.size() > MAX_IMAGES)
		throw std::runtime_error("Image pool maximum size exceeded");

	return &images[imageIndexMap[fileName]];
}

uint64_t ImagePool::getImageIndex(const std::string& fileName)
{
	if (imageIndexMap.count(fileName))
		return imageIndexMap[fileName];
	else
		throw std::runtime_error("Image index requested for non-existing image");
}

const std::vector<Image>& ImagePool::getImages()
{
	return images;
}

void ImagePool::clear()
{
	imageIndexMap.clear();
	images.clear();
}

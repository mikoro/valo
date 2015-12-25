// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Rendering/ImagePool.h"
#include "Rendering/Image.h"

using namespace Raycer;

std::map<std::string, uint64_t> ImagePool::imageIndexMap = std::map<std::string, uint64_t>();
std::vector<Imagef> ImagePool::images = std::vector<Imagef>();
bool ImagePool::initialized = false;

const Imagef* ImagePool::loadImage(const std::string& fileName, bool applyGamma)
{
	if (!initialized)
	{
		images.reserve(MAX_IMAGES);
		initialized = true;
	}

	if (!imageIndexMap.count(fileName))
	{
		images.push_back(Imagef(fileName));
		imageIndexMap[fileName] = images.size() - 1;

		if (applyGamma)
			images.back().applyFastGamma(2.2);
	}

	// the limit is arbitrary, increase it if it becomes a problem
	// idea is to prevent push_back from invalidating pointers
	if (images.size() > MAX_IMAGES)
		throw std::runtime_error("Image pool maximum size exceeded");

	return &images[imageIndexMap[fileName]];
}

uint64_t ImagePool::getImageIndex(const std::string& fileName)
{
	return imageIndexMap[fileName];
}

const std::vector<Imagef>& ImagePool::getImages()
{
	return images;
}

void ImagePool::clear()
{
	imageIndexMap.clear();
	images.clear();
}

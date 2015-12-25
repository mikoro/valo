// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/ImageTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Rendering/ImagePool.h"

using namespace Raycer;

void ImageTexture::initialize()
{
	image = ImagePool::loadImage(imageFilePath, applyGamma);
}

Color ImageTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)position;

	return image->getPixelBilinear(texcoord.x, texcoord.y).toColor();
}

double ImageTexture::getValue(const Vector2& texcoord, const Vector3& position) const
{
	(void)position;

	return double(image->getPixelBilinear(texcoord.x, texcoord.y).r);
}

const Imagef* ImageTexture::getImage() const
{
	return image;
}

uint64_t ImageTexture::getImagePoolIndex() const
{
	return ImagePool::getImageIndex(imageFilePath);
}

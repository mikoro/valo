// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "App.h"
#include "Core/Scene.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/ImageTexture.h"
#include "Utils/ImagePool.h"

using namespace Raycer;

void ImageTexture::initialize()
{
	image = App::getImagePool().load(imageFileName, applyGamma);
}

CUDA_CALLABLE Color ImageTexture::getColor(const Vector2& texcoord, const Vector3& position)
{
	(void)position;

	return image->getPixelBilinear(texcoord.x, texcoord.y);
}

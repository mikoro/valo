// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Scene.h"
#include "Core/ImagePool.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/ImageTexture.h"

using namespace Raycer;

void ImageTexture::initialize(Scene& scene)
{
	imageIndex = scene.imagePool.load(imageFileName, applyGamma);
}

CUDA_CALLABLE Color ImageTexture::getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position)
{
	(void)position;

	const Image& image = scene.imagePool.getImages()[imageIndex];
	return image.getPixelBilinear(texcoord.x, texcoord.y);
}

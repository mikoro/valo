// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/ImageTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Rendering/ImagePool.h"
#include "Scenes/Scene.h"

using namespace Raycer;

void ImageTexture::initialize(Scene& scene)
{
	image = scene.imagePool.getImage(imageFilePath, applyGamma);
}

Color ImageTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)position;

	return image->getPixelBilinear(texcoord.x, texcoord.y) * intensity;
}

float ImageTexture::getValue(const Vector2& texcoord, const Vector3& position) const
{
	(void)position;

	return image->getPixelBilinear(texcoord.x, texcoord.y).r * intensity;
}

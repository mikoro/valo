// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Raycer;

void Texture::initialize()
{
	switch (type)
	{
		case TextureType::CHECKER: checkerTexture.initialize(); break;
		case TextureType::IMAGE: imageTexture.initialize(); break;
		default: break;
	}
}

CUDA_CALLABLE Color Texture::getColor(const Vector2& texcoord, const Vector3& position)
{
	switch (type)
	{
		case TextureType::CHECKER: return checkerTexture.getColor(texcoord, position);
		case TextureType::IMAGE: return imageTexture.getColor(texcoord, position);
		default: return Color::black();
	}
}

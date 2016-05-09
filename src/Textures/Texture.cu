// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Valo;

Texture::Texture(TextureType type_) : type(type_)
{
}

void Texture::initialize(Scene& scene)
{
	switch (type)
	{
		case TextureType::IMAGE: imageTexture.initialize(scene); break;
		case TextureType::CHECKER: break;
		case TextureType::MARBLE: marbleTexture.initialize(); break;
		case TextureType::WOOD: woodTexture.initialize(); break;
		case TextureType::FIRE: fireTexture.initialize(); break;
		default: break;
	}
}

CUDA_CALLABLE Color Texture::getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
	switch (type)
	{
		case TextureType::IMAGE: return imageTexture.getColor(scene, texcoord, position);
		case TextureType::CHECKER: return checkerTexture.getColor(texcoord, position);
		case TextureType::MARBLE: return marbleTexture.getColor(texcoord, position);
		case TextureType::WOOD: return woodTexture.getColor(texcoord, position);
		case TextureType::FIRE: return fireTexture.getColor(texcoord, position);
		default: return Color::black();
	}
}

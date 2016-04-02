// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Textures/Texture.h"

using namespace Raycer;

void Texture::initialize(Scene& scene)
{
	switch (type)
	{
		case TextureType::CHECKER: break;
		case TextureType::IMAGE: imageTexture.initialize(scene); break;
		default: break;
	}
}

CUDA_CALLABLE Color Texture::getColor(const Scene& scene, const Vector2& texcoord, const Vector3& position)
{
	switch (type)
	{
		case TextureType::CHECKER: return checkerTexture.getColor(texcoord, position);
		case TextureType::IMAGE: return imageTexture.getColor(scene, texcoord, position);
		default: return Color::black();
	}
}

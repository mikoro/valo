// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/PerlinNoiseTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Scenes/Scene.h"

using namespace Raycer;

void PerlinNoiseTexture::initialize(Scene& scene)
{
	(void)scene;

	perlinNoise.initialize(seed);
}

Color PerlinNoiseTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	return baseColor * getValue(texcoord, position);
}

double PerlinNoiseTexture::getValue(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;

	if (isFbm)
		return perlinNoise.getFbmNoise(octaves, lacunarity, persistence, position.x * scale.x, position.y * scale.y, position.z * scale.z) * intensity;
	else
		return perlinNoise.getNoise(position.x * scale.x, position.y * scale.y, position.z * scale.z) * intensity;
}

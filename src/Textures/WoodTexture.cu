// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/WoodTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Color.h"

using namespace Valo;

void WoodTexture::initialize()
{
	noise.initialize(seed);
}

CUDA_CALLABLE Color WoodTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;

	// large scale pattern with sharp edges
	float n1 = noise.getFbmNoise(8, 2.0f, 0.3f, position.x * 2.0f * scale, position.y * 0.1f * scale, position.z * 2.0f * scale);
	n1 *= density;
	n1 -= int32_t(n1);
	n1 += 0.4f;
	n1 = MIN(n1, 1.0f);

	// subtle bumpiness
	float n2 = (1.0f - 1.0f / bumpiness) + noise.getNoise(position.x * 16.0f * scale, position.y * 16.0f * scale, position.z * 16.0f * scale) / bumpiness;

	// subtle streaks
	float n3 = noise.getFbmNoise(4, 2.0f, 0.01f, position.x * 200.0f * scale, position.y * 1.0f * scale, position.z * 2.0f * scale);
	n3 = 1.0f - n3 * n3 * n3 * n3;
	n3 = 0.75f + n3 / 4.0f;

	return woodColor * (n1 * n2 * n3);
}

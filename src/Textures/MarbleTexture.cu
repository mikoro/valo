// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/MarbleTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Color.h"

using namespace Raycer;

void MarbleTexture::initialize()
{
	noise.initialize(seed);
}

CUDA_CALLABLE Color MarbleTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;

	float n1 = std::abs(std::cos(position.x * density + noise.getFbmNoise(8, 2.0f, 0.5f, position.x * 2.0f, position.y * 2.0f, position.z * 2.0f) * swirliness));
	n1 = (1.0f - n1) / transparency;

	Color streakColor1(streakColor);
	streakColor1.a = n1;

	return Color::alphaBlend(marbleColor, streakColor1);
}

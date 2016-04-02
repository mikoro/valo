// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/CheckerTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"

using namespace Raycer;

CUDA_CALLABLE Color CheckerTexture::getColor(const Vector2& texcoord, const Vector3& position)
{
	(void)position;

	if (stripeMode)
	{
		if (texcoord.x < stripeWidth || texcoord.y < stripeWidth || texcoord.x >(1.0f - stripeWidth) || texcoord.y >(1.0f - stripeWidth))
			return color1;

		return color2;
	}

	if (texcoord.x < 0.5f)
		return ((texcoord.y < 0.5f) ? color1 : color2);

	return ((texcoord.y < 0.5f) ? color2 : color1);
}

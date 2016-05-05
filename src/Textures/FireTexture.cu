// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/FireTexture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Color.h"

using namespace Raycer;

void FireTexture::initialize()
{
	noise.initialize(seed);

	gradient.addSegment(Color(0, 0, 0), Color(0, 0, 0), 50);
	gradient.addSegment(Color(0, 0, 0), Color(5, 5, 5), 4);
	gradient.addSegment(Color(5, 5, 5), Color(255, 0, 0), 6);
	gradient.addSegment(Color(255, 0, 0), Color(255, 255, 0), 30);
	gradient.addSegment(Color(255, 255, 0), Color(255, 255, 255), 10);

	gradient.commit();
}

CUDA_CALLABLE Color FireTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;

	float n1 = noise.getFbmNoise(8, 2.0f, 0.75f, position.x * scale * 4.0f, position.y * scale * 4.0f, position.z * scale * 4.0f) / 3.0f;
	n1 = std::max(0.0f, std::min(n1, 1.0f));
	
	return gradient.getColor(n1);
}

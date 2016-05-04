// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Core/Image.h"
#include "Core/PerlinNoise.h"
#include "Math/Color.h"

using namespace Raycer;

TEST_CASE("Perlin noise functionality", "[perlin]")
{
	Image image(500, 500);
	PerlinNoise noise;
	noise.initialize(12345);

	for (uint32_t y = 0; y < 500; ++y)
	{
		for (uint32_t x = 0; x < 500; ++x)
		{
			float n = noise.getNoise(x * 0.01f, y * 0.01f, 0.0f);
			image.setPixel(x, y, Color(n, n, n));
		}
	}

	image.save("perlin_noise.png");
}

#endif

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Core/Image.h"
#include "Core/Scene.h"
#include "Textures/Texture.h"

using namespace Raycer;

TEST_CASE("Texture functionality", "[texture]")
{
	Image image1(500, 500);
	Image image2(500, 500);
	Image image3(500, 500);
	Image image4(500, 500);

	Texture checkerTexture(TextureType::CHECKER);
	Texture marbleTexture(TextureType::MARBLE);
	Texture woodTexture(TextureType::WOOD);
	Texture fireTexture(TextureType::FIRE);

	marbleTexture.marbleTexture.seed = 32434;
	woodTexture.woodTexture.seed = 45343;
	fireTexture.fireTexture.seed = 65432;

	Scene scene;
	marbleTexture.initialize(scene);
	woodTexture.initialize(scene);
	fireTexture.initialize(scene);

	for (uint32_t y = 0; y < 500; ++y)
	{
		for (uint32_t x = 0; x < 500; ++x)
		{
			image1.setPixel(x, y, checkerTexture.getColor(scene, Vector2(x * 0.1f - floor(x * 0.1f), y * 0.1f - floor(y * 0.1f)), Vector3()));
			image2.setPixel(x, y, marbleTexture.getColor(scene, Vector2(), Vector3(x * 0.01f, y * 0.01f, 1.0f)));
			image3.setPixel(x, y, woodTexture.getColor(scene, Vector2(), Vector3(x * 0.01f, y * 0.01f, 0.0f)));
			image4.setPixel(x, y, fireTexture.getColor(scene, Vector2(), Vector3(x * 0.01f, y * 0.01f, 0.0f)));
		}
	}

	image1.save("texture_checker.png");
	image2.save("texture_marble.png");
	image3.save("texture_wood.png");
	image4.save("texture_fire.png");
}

#endif

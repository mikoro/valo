// Copyright � 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "catch/catch.hpp"

#include "Rendering/Image.h"
#include "Math/Color.h"

using namespace Raycer;

TEST_CASE("Image functionality", "[image]")
{
	Image image(100, 100);

	for (int y = 0; y < 100; ++y)
	{
		for (int x = 0; x < 100; ++x)
		{
			if (y < 50)
				image.setPixel(x, y, Color(10, 100, 200, 255));
			else
				image.setPixel(x, y, Color(255, 0, 0, 128));
		}
	}

	image.saveAs("image1.png");
	image.saveAs("image1.tga");
	image.saveAs("image1.bmp");
	image.saveAs("image1.hdr");

	image.load("image1.png");
	image.saveAs("image2.png");

	image.load("image1.tga");
	image.saveAs("image2.tga");

	image.load("image1.bmp");
	image.saveAs("image2.bmp");

	image.load("image1.hdr");
	image.saveAs("image2.hdr");
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Core/Image.h"
#include "Filters/Filter.h"
#include "Math/Color.h"

using namespace Raycer;

TEST_CASE("Image save/load functionality", "[image]")
{
	Image image(100, 100);

	for (uint32_t y = 0; y < 100; ++y)
	{
		for (uint32_t x = 0; x < 100; ++x)
		{
			if (y < 50)
				image.setPixel(x, y, Color(10, 100, 200, 255));
			else
				image.setPixel(x, y, Color(255, 0, 0, 128));
		}
	}

	image.save("image1.png");
	image.save("image1.tga");
	image.save("image1.bmp");
	image.save("image1.hdr");

	image.load("image1.png");
	image.save("image2.png");

	image.load("image1.tga");
	image.save("image2.tga");

	image.load("image1.bmp");
	image.save("image2.bmp");

	image.load("image1.hdr");
	image.save("image2.hdr");
}

TEST_CASE("Image interpolation functionality", "[image]")
{
	Image image1(101, 101);
	Image image2(201, 201);
	Image image3(201, 201);
	Image image4(201, 201);

	image1.fillWithTestPattern();

	Filter filter(FilterType::MITCHELL);

	for (uint32_t y = 0; y <= 200; ++y)
	{
		for (uint32_t x = 0; x <= 200; ++x)
		{
			float u = x / 200.0f;
			float v = y / 200.0f;

			Color c = image1.getPixelNearest(u, v);
			image2.setPixel(x, y, c);

			c = image1.getPixelBilinear(u, v);
			image3.setPixel(x, y, c);

			c = image1.getPixelBicubic(u, v, filter);
			image4.setPixel(x, y, c);
		}
	}

	image1.save("interpolation_original.png");
	image2.save("interpolation_nearest.png");
	image3.save("interpolation_bilinear.png");
	image4.save("interpolation_bicubic.png");
}

#endif

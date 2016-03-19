// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Tonemappers/Tonemapper.h"
#include "Core/Image.h"

using namespace Raycer;

void Tonemapper::apply(const Image& inputImage, Image& outputImage)
{
	assert(inputImage.getLength() == outputImage.getLength());

	switch (type)
	{
		case TonemapperType::PASSTHROUGH: passthroughTonemapper.apply(inputImage, outputImage); break;
		case TonemapperType::LINEAR: linearTonemapper.apply(inputImage, outputImage); break;
		case TonemapperType::SIMPLE: simpleTonemapper.apply(inputImage, outputImage); break;
		case TonemapperType::REINHARD: reinhardTonemapper.apply(inputImage, outputImage); break;
		default: break;
	}
}

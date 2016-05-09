// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tonemappers/Tonemapper.h"
#include "Core/Image.h"

using namespace Valo;

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

std::string Tonemapper::getName() const
{
	switch (type)
	{
		case TonemapperType::PASSTHROUGH: return "passthrough";
		case TonemapperType::LINEAR: return "linear";
		case TonemapperType::SIMPLE: return "simple";
		case TonemapperType::REINHARD: return "reinhard";
		default: return "unknown";
	}
}

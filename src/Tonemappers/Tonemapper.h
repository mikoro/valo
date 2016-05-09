// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#ifdef PASSTHROUGH
#undef PASSTHROUGH
#endif

#include "Tonemappers/PassthroughTonemapper.h"
#include "Tonemappers/LinearTonemapper.h"
#include "Tonemappers/SimpleTonemapper.h"
#include "Tonemappers/ReinhardTonemapper.h"

namespace Valo
{
	class Image;

	enum class TonemapperType { PASSTHROUGH, LINEAR, SIMPLE, REINHARD };

	class Tonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);

		std::string getName() const;

		TonemapperType type = TonemapperType::LINEAR;

		PassthroughTonemapper passthroughTonemapper;
		LinearTonemapper linearTonemapper;
		SimpleTonemapper simpleTonemapper;
		ReinhardTonemapper reinhardTonemapper;
	};
}

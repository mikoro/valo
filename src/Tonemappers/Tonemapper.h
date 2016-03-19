// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#ifdef PASSTHROUGH
#undef PASSTHROUGH
#endif

#include "cereal/cereal.hpp"

#include "Tonemappers/PassthroughTonemapper.h"
#include "Tonemappers/LinearTonemapper.h"
#include "Tonemappers/SimpleTonemapper.h"
#include "Tonemappers/ReinhardTonemapper.h"

namespace Raycer
{
	class Image;

	enum class TonemapperType { PASSTHROUGH, LINEAR, SIMPLE, REINHARD };

	class Tonemapper
	{
	public:

		void apply(const Image& inputImage, Image& outputImage);

		TonemapperType type = TonemapperType::PASSTHROUGH;

		PassthroughTonemapper passthroughTonemapper;
		LinearTonemapper linearTonemapper;
		SimpleTonemapper simpleTonemapper;
		ReinhardTonemapper reinhardTonemapper;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(type),
				//CEREAL_NVP(passthroughTonemapper),
				CEREAL_NVP(linearTonemapper),
				CEREAL_NVP(simpleTonemapper),
				CEREAL_NVP(reinhardTonemapper));
		}
	};
}

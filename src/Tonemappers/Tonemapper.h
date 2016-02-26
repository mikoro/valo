// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#ifdef PASSTHROUGH
#undef PASSTHROUGH
#endif

#include "Rendering/Image.h"

namespace Raycer
{
	class Scene;

	enum class TonemapperType { PASSTHROUGH, LINEAR, SIMPLE, REINHARD };

	class Tonemapper
	{
	public:

		virtual ~Tonemapper() {}

		virtual void apply(const Scene& scene, const Image& inputImage, Image& outputImage) = 0;

		static std::unique_ptr<Tonemapper> getTonemapper(TonemapperType type);
	};
}

// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Tonemappers/Tonemapper.h"

namespace Raycer
{
	class Scene;

	class SimpleTonemapper : public Tonemapper
	{
	public:

		void apply(const Scene& scene, const Image& inputImage, Image& outputImage) override;
	};
}

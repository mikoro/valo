// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Intersection;
	class Random;
	class Vector3;

	class Light
	{
	public:

		virtual ~Light() {}

		virtual void initialize() = 0;
		virtual bool hasDirection() const = 0;

		virtual Color getColor(const Scene& scene, const Intersection& intersection, Random& random) const = 0;
		virtual Vector3 getDirection(const Intersection& intersection) const = 0;
	};
}

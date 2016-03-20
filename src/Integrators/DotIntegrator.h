// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Color;
	class Scene;
	class Ray;
	class Random;

	class DotIntegrator
	{
	public:

		Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random);

		uint32_t getSampleCount() const;
	};
}

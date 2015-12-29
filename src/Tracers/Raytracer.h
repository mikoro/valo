// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Tracers/Tracer.h"

namespace Raycer
{
	class Scene;
	class Vector3;
	class Ray;
	class Intersection;

	class Raytracer : public Tracer
	{
	protected:

		Color trace(const Scene& scene, const Ray& ray, Random& random) override;

	private:

		Color traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random);

		void calculateNormalMapping(Intersection& intersection);
		void calculateRayReflectanceAndTransmittance(const Intersection& intersection, double& rayReflectance, double& rayTransmittance);
		Color calculateReflectedColor(const Scene& scene, const Intersection& intersection, double rayReflectance, uint64_t iteration, Random& random);
		Color calculateTransmittedColor(const Scene& scene, const Intersection& intersection, double rayTransmittance, uint64_t iteration, Random& random);
		Color calculateMaterialColor(const Scene& scene, const Intersection& intersection, Random& random);
	};
}

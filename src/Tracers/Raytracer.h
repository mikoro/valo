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
	public:

		uint64_t getPixelSampleCount(const Scene& scene) const override;
		uint64_t getSamplesPerPixel(const Scene& scene) const override;

	protected:

		void trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& pathCount) override;

	private:

		void generateMultiSamples(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random);
		Color generateCameraSamples(const Scene& scene, const Vector2& pixelCenter, Random& random);

		Color traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random);

		void calculateRayReflectanceAndTransmittance(const Intersection& intersection, double& rayReflectance, double& rayTransmittance);
		Color calculateReflectedColor(const Scene& scene, const Intersection& intersection, double rayReflectance, uint64_t iteration, Random& random);
		Color calculateTransmittedColor(const Scene& scene, const Intersection& intersection, double rayTransmittance, uint64_t iteration, Random& random);
		Color calculateMaterialColor(const Scene& scene, const Intersection& intersection, Random& random);
	};
}

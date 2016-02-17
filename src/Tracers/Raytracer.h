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

		void trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount) override;

	private:

		void generateMultiSamples(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount);
		Color generateCameraSamples(const Scene& scene, const Vector2& pixelCenter, Random& random, uint64_t& rayCount);

		Color traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random, uint64_t& rayCount);

		void calculateRayReflectanceAndTransmittance(const Intersection& intersection, float& rayReflectance, float& rayTransmittance);
		Color calculateReflectedColor(const Scene& scene, const Intersection& intersection, float rayReflectance, uint64_t iteration, Random& random, uint64_t& rayCount);
		Color calculateTransmittedColor(const Scene& scene, const Intersection& intersection, float rayTransmittance, uint64_t iteration, Random& random, uint64_t& rayCount);
		Color calculateMaterialColor(const Scene& scene, const Intersection& intersection, Random& random);
	};
}

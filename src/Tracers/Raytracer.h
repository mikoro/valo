// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Tracers/Tracer.h"

namespace Raycer
{
	class Scene;
	class Vector3;
	class Ray;
	struct Intersection;
	struct Light;
	struct DirectionalLight;
	struct PointLight;

	class Raytracer : public Tracer
	{
	protected:

		Color trace(const Scene& scene, const Ray& ray, Random& random) override;

	private:

		Color traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random);

		void calculateNormalMapping(Intersection& intersection);
		void calculateRayReflectanceAndTransmittance(const Ray& ray, const Intersection& intersection, double& rayReflectance, double& rayTransmittance);
		Color calculateReflectedColor(const Scene& scene, const Ray& ray, const Intersection& intersection, double rayReflectance, uint64_t iteration, Random& random);
		Color calculateTransmittedColor(const Scene& scene, const Ray& ray, const Intersection& intersection, double rayTransmittance, uint64_t iteration, Random& random);
		Color calculateLightColor(const Scene& scene, const Ray& ray, const Intersection& intersection, Random& random);
		Color calculatePhongShadingColor(const Vector3& normal, const Vector3& directionToLight, const Vector3& directionToCamera, const Light& light, const Color& diffuseReflectance, const Color& specularReflectance, double shininess);
		double calculateShadowAmount(const Scene& scene, const Ray& ray, const Intersection& intersection, const DirectionalLight& light);
		double calculateShadowAmount(const Scene& scene, const Ray& ray, const Intersection& intersection, const PointLight& light, Random& random);
	};
}

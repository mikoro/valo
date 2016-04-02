// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/Integrator.h"
#include "Materials/Material.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Color Integrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random)
{
	switch (type)
	{
		case IntegratorType::DOT: return dotIntegrator.calculateRadiance(scene, viewRay, random);
		case IntegratorType::PATH: return pathIntegrator.calculateRadiance(scene, viewRay, random);
		default: return Color::black();
	}
}

uint32_t Integrator::getSampleCount() const
{
	switch (type)
	{
		case IntegratorType::DOT: return dotIntegrator.getSampleCount();
		case IntegratorType::PATH: return pathIntegrator.getSampleCount();
		default: return 0;
	}
}

std::string Integrator::getName() const
{
	switch (type)
	{
		case IntegratorType::DOT: return "dot";
		case IntegratorType::PATH: return "path";
		default: return "unknown";
	}
}

CUDA_CALLABLE Color Integrator::calculateDirectLight(const Scene& scene, const Intersection& intersection, const Vector3& in, Random& random)
{
	if (scene.getEmissiveTrianglesCount() == 0)
		return Color(0.0f, 0.0f, 0.0f);

	Triangle& triangle = scene.getEmissiveTriangles()[random.getUint32(0, scene.getEmissiveTrianglesCount() - 1)];
	Intersection triangleIntersection = triangle.getRandomIntersection(scene, random);
	Vector3 intersectionToTriangle = triangleIntersection.position - intersection.position;
	float triangleDistance2 = intersectionToTriangle.lengthSquared();
	float triangleDistance = sqrt(triangleDistance2);
	Vector3 out = intersectionToTriangle / triangleDistance;

	Ray visibilityRay;
	visibilityRay.origin = intersection.position;
	visibilityRay.direction = out;
	visibilityRay.minDistance = scene.general.rayMinDistance;
	visibilityRay.maxDistance = triangleDistance - scene.general.rayMinDistance;
	visibilityRay.isVisibilityRay = true;
	visibilityRay.precalculate();

	Intersection visibilityIntersection;

	if (scene.intersect(visibilityRay, visibilityIntersection))
		return Color(0.0f, 0.0f, 0.0f);

	float cosine1 = intersection.normal.dot(out);
	float cosine2 = out.dot(-triangle.normal);

	if (cosine1 < 0.0f || cosine2 < 0.0f)
		return Color(0.0f, 0.0f, 0.0f);

	float probability1 = 1.0f / float(scene.getEmissiveTrianglesCount());
	float probability2 = 1.0f / triangle.area;

	Material& triangleMaterial = scene.getMaterial(triangle.materialIndex);
	Material& intersectionMaterial = scene.getMaterial(intersection.materialIndex);

	Color emittance = triangleMaterial.getEmittance(scene, triangleIntersection.texcoord, triangleIntersection.position);
	Color brdf = intersectionMaterial.getBrdf(scene, intersection, in, out);

	return emittance * brdf * cosine1 * cosine2 * (1.0f / triangleDistance2) / (probability1 * probability2);
}

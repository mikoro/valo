// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Integrators/Integrator.h"
#include "Materials/Material.h"
#include "Textures/Texture.h"
#include "Utils/Random.h"

using namespace Raycer;

Color Integrator::calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random)
{
	switch (type)
	{
		case IntegratorType::DOT: return dotIntegrator.calculateRadiance(scene, viewRay, random);
		case IntegratorType::PATH: return pathIntegrator.calculateRadiance(scene, viewRay, random);
		default: return Color::BLACK;
	}
}

Color Integrator::calculateDirectLight(const Scene& scene, const Intersection& intersection, Random& random)
{
	if (scene.emissiveTrianglesCount == 0)
		return Color(0.0f, 0.0f, 0.0f);

	Triangle* emitter = &scene.emissiveTrianglesPtr[random.getUint64(0, scene.emissiveTrianglesCount - 1)];
	Intersection emitterIntersection = emitter->getRandomIntersection(random);
	Vector3 intersectionToEmitter = emitterIntersection.position - intersection.position;
	float emitterDistance2 = intersectionToEmitter.lengthSquared();
	float emitterDistance = sqrt(emitterDistance2);
	Vector3 sampleDirection = intersectionToEmitter / emitterDistance;

	Ray shadowRay;
	shadowRay.origin = intersection.position;
	shadowRay.direction = sampleDirection;
	shadowRay.minDistance = scene.general.rayMinDistance;
	shadowRay.maxDistance = emitterDistance - scene.general.rayMinDistance;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.precalculate();

	Intersection shadowIntersection;
	scene.intersect(shadowRay, shadowIntersection);

	if (shadowIntersection.wasFound)
		return Color(0.0f, 0.0f, 0.0f);

	float cosine1 = intersection.normal.dot(sampleDirection);
	float cosine2 = sampleDirection.dot(-emitter->normal);

	if (cosine1 < 0.0f || cosine2 < 0.0f)
		return Color(0.0f, 0.0f, 0.0f);

	float probability1 = 1.0f / float(scene.emissiveTrianglesCount);
	float probability2 = 1.0f / emitter->getArea();

	Color emittance = emitter->material->getEmittance(emitterIntersection.texcoord, emitterIntersection.position);
	Color intersectionBrdf = intersection.material->getBrdf(intersection, sampleDirection);

	return emittance * intersectionBrdf * cosine1 * cosine2 * (1.0f / emitterDistance2) / (probability1 * probability2);
}

void Integrator::calculateNormalMapping(Intersection& intersection)
{
	Color normalColor = intersection.material->normalTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b * 2.0f - 1.0f);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

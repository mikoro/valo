// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Raytracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Textures/Texture.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Tracing/ONB.h"

using namespace Raycer;

Color Raytracer::trace(const Scene& scene, const Ray& ray, Random& random)
{
	Intersection intersection;
	return traceRecursive(scene, ray, intersection, 0, random);
}

Color Raytracer::traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random)
{
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return scene.general.backgroundColor;

	Material* material = intersection.material;

	if (material->skipLighting)
		return material->getDiffuseReflectance(intersection);

	if (scene.general.enableNormalMapping && material->normalMapTexture != nullptr)
		calculateNormalMapping(intersection);

	double rayReflectance, rayTransmittance;

	calculateRayReflectanceAndTransmittance(intersection, rayReflectance, rayTransmittance);

	Color reflectedColor, transmittedColor;

	if (rayReflectance > 0.0 && iteration < scene.raytracing.maxRayIterations)
		reflectedColor = calculateReflectedColor(scene, intersection, rayReflectance, iteration, random);

	if (rayTransmittance > 0.0 && iteration < scene.raytracing.maxRayIterations)
		transmittedColor = calculateTransmittedColor(scene, intersection, rayTransmittance, iteration, random);

	Color materialColor = calculateMaterialColor(scene, intersection, random);

	return materialColor + reflectedColor + transmittedColor;
}

void Raytracer::calculateNormalMapping(Intersection& intersection)
{
	Color normalColor = intersection.material->normalMapTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0 - 1.0, normalColor.g * 2.0 - 1.0, normalColor.b);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

void Raytracer::calculateRayReflectanceAndTransmittance(const Intersection& intersection, double& rayReflectance, double& rayTransmittance)
{
	Material* material = intersection.material;

	double fresnelReflectance = 1.0;
	double fresnelTransmittance = 1.0;

	if (material->fresnelReflection)
	{
		double cosine = intersection.rayDirection.dot(intersection.normal);
		bool isOutside = cosine < 0.0;
		double n1 = isOutside ? 1.0 : material->refractiveIndex;
		double n2 = isOutside ? material->refractiveIndex : 1.0;
		double rf0 = (n2 - n1) / (n2 + n1);
		rf0 = rf0 * rf0;
		fresnelReflectance = rf0 + (1.0 - rf0) * pow(1.0 - std::abs(cosine), 5.0);
		fresnelTransmittance = 1.0 - fresnelReflectance;
	}

	rayReflectance = material->rayReflectance * fresnelReflectance;
	rayTransmittance = material->rayTransmittance * fresnelTransmittance;
}

Color Raytracer::calculateReflectedColor(const Scene& scene, const Intersection& intersection, double rayReflectance, uint64_t iteration, Random& random)
{
	Material* material = intersection.material;

	Vector3 reflectionDirection = intersection.rayDirection + 2.0 * -intersection.rayDirection.dot(intersection.normal) * intersection.normal;
	reflectionDirection.normalize();

	Color reflectedColor;
	bool isOutside = intersection.rayDirection.dot(intersection.normal) < 0.0;

	Ray reflectedRay;
	Intersection reflectedIntersection;

	reflectedRay.origin = intersection.position;
	reflectedRay.direction = reflectionDirection;
	reflectedRay.minDistance = scene.general.rayMinDistance;
	reflectedRay.precalculate();

	reflectedColor = traceRecursive(scene, reflectedRay, reflectedIntersection, iteration + 1, random) * rayReflectance;

	// only attenuate if ray has traveled inside a primitive
	if (!isOutside && reflectedIntersection.wasFound && material->attenuating)
	{
		double a = exp(-material->attenuationFactor * reflectedIntersection.distance);
		reflectedColor = Color::lerp(material->attenuationColor, reflectedColor, a);
	}

	return reflectedColor;
}

Color Raytracer::calculateTransmittedColor(const Scene& scene, const Intersection& intersection, double rayTransmittance, uint64_t iteration, Random& random)
{
	Material* material = intersection.material;

	double cosine1 = intersection.rayDirection.dot(intersection.normal);
	bool isOutside = cosine1 < 0.0;
	double n1 = isOutside ? 1.0 : material->refractiveIndex;
	double n2 = isOutside ? material->refractiveIndex : 1.0;
	double n3 = n1 / n2;
	double cosine2 = 1.0 - (n3 * n3) * (1.0 - cosine1 * cosine1);

	Color transmittedColor;

	// total internal reflection -> no transmission
	if (cosine2 <= 0.0)
		return transmittedColor;

	Vector3 transmissionDirection = intersection.rayDirection * n3 + (std::abs(cosine1) * n3 - sqrt(cosine2)) * intersection.normal;
	transmissionDirection.normalize();

	Ray transmittedRay;
	Intersection transmittedIntersection;

	transmittedRay.origin = intersection.position;
	transmittedRay.direction = transmissionDirection;
	transmittedRay.minDistance = scene.general.rayMinDistance;
	transmittedRay.precalculate();

	transmittedColor = traceRecursive(scene, transmittedRay, transmittedIntersection, iteration + 1, random) * rayTransmittance;

	// only attenuate if ray has traveled inside a primitive
	if (isOutside && transmittedIntersection.wasFound && material->attenuating)
	{
		double a = exp(-material->attenuationFactor * transmittedIntersection.distance);
		transmittedColor = Color::lerp(material->attenuationColor, transmittedColor, a);
	}

	return transmittedColor;
}

Color Raytracer::calculateMaterialColor(const Scene& scene, const Intersection& intersection, Random& random)
{
	Material* material = intersection.material;

	Color materialColor;

	for (Light* light : scene.lightsList)
		materialColor += material->getColor(scene, intersection, *light, random);

	return materialColor;
}

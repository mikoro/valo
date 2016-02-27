// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Raytracer.h"
#include "Tracers/TracerCommon.h"
#include "Tracing/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Rendering/Film.h"

using namespace Raycer;

uint64_t Raytracer::getPixelSampleCount(const Scene& scene) const
{
	(void)scene;

	return 1;
}

uint64_t Raytracer::getSamplesPerPixel(const Scene& scene) const
{
	return scene.raytracing.multiSampleCountSqrt *
		scene.raytracing.multiSampleCountSqrt *
		scene.raytracing.cameraSampleCountSqrt *
		scene.raytracing.cameraSampleCountSqrt;
}

void Raytracer::trace(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount, uint64_t& pathCount)
{
	pathCount = 0;

	generateMultiSamples(scene, film, pixelCenter, pixelIndex, random, rayCount);
}

void Raytracer::generateMultiSamples(const Scene& scene, Film& film, const Vector2& pixelCenter, uint64_t pixelIndex, Random& random, uint64_t& rayCount)
{
	assert(scene.raytracing.multiSampleCountSqrt >= 1);

	if (scene.raytracing.multiSampleCountSqrt == 1)
	{
		Color pixelColor = generateCameraSamples(scene, pixelCenter, random, rayCount);
		film.addSample(pixelIndex, pixelColor, 1.0f);
		return;
	}

	Sampler* sampler = samplers[scene.raytracing.multiSamplerType].get();
	Filter* filter = filters[scene.raytracing.multiSamplerFilterType].get();

	uint64_t permutation = random.getUint64();
	uint64_t n = scene.raytracing.multiSampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector2 sampleOffset = sampler->getSquareSample(x, y, n, n, permutation, random);
			sampleOffset = (sampleOffset - Vector2(0.5f, 0.5f)) * 2.0f * filter->getRadius();
			Color sampledPixelColor = generateCameraSamples(scene, pixelCenter + sampleOffset, random, rayCount);
			film.addSample(pixelIndex, sampledPixelColor, filter->getWeight(sampleOffset));
		}
	}
}

Color Raytracer::generateCameraSamples(const Scene& scene, const Vector2& pixelCenter, Random& random, uint64_t& rayCount)
{
	assert(scene.raytracing.cameraSampleCountSqrt >= 1);

	bool isOffLens;
	Ray ray = scene.camera.getRay(pixelCenter, isOffLens);

	if (isOffLens)
		return scene.general.offLensColor;

	if (scene.raytracing.cameraSampleCountSqrt == 1)
	{
		Intersection intersection;
		return traceRecursive(scene, ray, intersection, 0, random, rayCount);
	}

	Vector3 cameraPosition = scene.camera.position;
	Vector3 cameraRight = scene.camera.getRight();
	Vector3 cameraUp = scene.camera.getUp();
	Vector3 focalPoint = ray.origin + ray.direction * scene.camera.focalDistance;

	Sampler* sampler = samplers[scene.raytracing.cameraSamplerType].get();

	uint64_t permutation = random.getUint64();
	uint64_t n = scene.raytracing.cameraSampleCountSqrt;

	Color sampledPixelColor;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector2 discCoordinate = sampler->getDiscSample(x, y, n, n, permutation, random);

			Ray sampleRay;
			Intersection sampleIntersection;

			sampleRay.origin = cameraPosition + ((discCoordinate.x * scene.camera.apertureSize) * cameraRight + (discCoordinate.y * scene.camera.apertureSize) * cameraUp);
			sampleRay.direction = (focalPoint - sampleRay.origin).normalized();
			sampleRay.precalculate();

			sampledPixelColor += traceRecursive(scene, ray, sampleIntersection, 0, random, rayCount);
		}
	}

	return sampledPixelColor / (float(n) * float(n));
}

Color Raytracer::traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, Random& random, uint64_t& rayCount)
{
	++rayCount;

	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return scene.general.backgroundColor;

	Material* material = intersection.material;

	if (material->skipLighting)
		return material->getReflectance(intersection);

	if (scene.general.normalMapping && material->normalTexture != nullptr)
		TracerCommon::calculateNormalMapping(intersection);

	float rayReflectance, rayTransmittance;

	calculateRayReflectanceAndTransmittance(intersection, rayReflectance, rayTransmittance);

	Color reflectedColor, transmittedColor;

	if (rayReflectance > 0.0f && iteration < scene.raytracing.maxIterationDepth)
		reflectedColor = calculateReflectedColor(scene, intersection, rayReflectance, iteration, random, rayCount);

	if (rayTransmittance > 0.0f && iteration < scene.raytracing.maxIterationDepth)
		transmittedColor = calculateTransmittedColor(scene, intersection, rayTransmittance, iteration, random, rayCount);

	Color materialColor = calculateMaterialColor(scene, intersection, random);

	return materialColor + reflectedColor + transmittedColor;
}

void Raytracer::calculateRayReflectanceAndTransmittance(const Intersection& intersection, float& rayReflectance, float& rayTransmittance)
{
	Material* material = intersection.material;

	float fresnelReflectance = 1.0f;
	float fresnelTransmittance = 1.0f;

	if (material->fresnelReflection)
	{
		float cosine = intersection.rayDirection.dot(intersection.normal);
		bool isOutside = cosine < 0.0f;
		float n1 = isOutside ? 1.0f : material->refractiveIndex;
		float n2 = isOutside ? material->refractiveIndex : 1.0f;
		float rf0 = (n2 - n1) / (n2 + n1);
		rf0 = rf0 * rf0;
		fresnelReflectance = rf0 + (1.0f - rf0) * std::pow(1.0f - std::abs(cosine), 5.0f);
		fresnelTransmittance = 1.0f - fresnelReflectance;
	}

	rayReflectance = material->rayReflectance * fresnelReflectance;
	rayTransmittance = material->rayTransmittance * fresnelTransmittance;
}

Color Raytracer::calculateReflectedColor(const Scene& scene, const Intersection& intersection, float rayReflectance, uint64_t iteration, Random& random, uint64_t& rayCount)
{
	Material* material = intersection.material;

	Vector3 reflectionDirection = intersection.rayDirection + 2.0f * -intersection.rayDirection.dot(intersection.normal) * intersection.normal;
	reflectionDirection.normalize();

	Color reflectedColor;
	bool isOutside = intersection.rayDirection.dot(intersection.normal) < 0.0f;

	Ray reflectedRay;
	Intersection reflectedIntersection;

	reflectedRay.origin = intersection.position;
	reflectedRay.direction = reflectionDirection;
	reflectedRay.minDistance = scene.general.rayMinDistance;
	reflectedRay.precalculate();

	reflectedColor = traceRecursive(scene, reflectedRay, reflectedIntersection, iteration + 1, random, rayCount) * rayReflectance;

	// only attenuate if ray has traveled inside
	if (!isOutside && reflectedIntersection.wasFound && material->attenuating)
	{
		float a = std::exp(-material->attenuationFactor * reflectedIntersection.distance);
		reflectedColor = Color::lerp(material->attenuationColor, reflectedColor, a);
	}

	return reflectedColor;
}

Color Raytracer::calculateTransmittedColor(const Scene& scene, const Intersection& intersection, float rayTransmittance, uint64_t iteration, Random& random, uint64_t& rayCount)
{
	Material* material = intersection.material;

	float cosine1 = intersection.rayDirection.dot(intersection.normal);
	bool isOutside = cosine1 < 0.0f;
	float n1 = isOutside ? 1.0f : material->refractiveIndex;
	float n2 = isOutside ? material->refractiveIndex : 1.0f;
	float n3 = n1 / n2;
	float cosine2 = 1.0f - (n3 * n3) * (1.0f - cosine1 * cosine1);

	Color transmittedColor;

	// total internal reflection -> no transmission
	if (cosine2 <= 0.0f)
		return transmittedColor;

	Vector3 transmissionDirection = intersection.rayDirection * n3 + (std::abs(cosine1) * n3 - std::sqrt(cosine2)) * intersection.normal;
	transmissionDirection.normalize();

	Ray transmittedRay;
	Intersection transmittedIntersection;

	transmittedRay.origin = intersection.position;
	transmittedRay.direction = transmissionDirection;
	transmittedRay.minDistance = scene.general.rayMinDistance;
	transmittedRay.precalculate();

	transmittedColor = traceRecursive(scene, transmittedRay, transmittedIntersection, iteration + 1, random, rayCount) * rayTransmittance;

	// only attenuate if ray has traveled inside
	if (isOutside && transmittedIntersection.wasFound && material->attenuating)
	{
		float a = std::exp(-material->attenuationFactor * transmittedIntersection.distance);
		transmittedColor = Color::lerp(material->attenuationColor, transmittedColor, a);
	}

	return transmittedColor;
}

Color Raytracer::calculateMaterialColor(const Scene& scene, const Intersection& intersection, Random& random)
{
	if (intersection.hasColor)
		return intersection.color;

	if (scene.general.normalVisualization)
		TracerCommon::calculateNormalColor(intersection.normal);

	Material* material = intersection.material;

	Color materialColor;

	for (Light* light : scene.lightsList)
		materialColor += material->getColor(scene, intersection, *light, random);

	return materialColor;
}

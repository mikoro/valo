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

CUDA_CALLABLE Color Integrator::calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const
{
	switch (type)
	{
		case IntegratorType::PATH: return pathIntegrator.calculateLight(scene, intersection, ray, random);
		case IntegratorType::DOT: return dotIntegrator.calculateLight(scene, intersection, ray, random);
		case IntegratorType::AMBIENT_OCCLUSION: return aoIntegrator.calculateLight(scene, intersection, ray, random);
		case IntegratorType::DIRECT_LIGHT: return directIntegrator.calculateLight(scene, intersection, ray, random);
		default: return Color::black();
	}
}

std::string Integrator::getName() const
{
	switch (type)
	{
		case IntegratorType::PATH: return "path";
		case IntegratorType::DOT: return "dot";
		case IntegratorType::AMBIENT_OCCLUSION: return "ao";
		case IntegratorType::DIRECT_LIGHT: return "direct";
		default: return "unknown";
	}
}

CUDA_CALLABLE Intersection Integrator::getRandomEmissiveIntersection(const Scene& scene, Random& random)
{
	const Triangle& triangle = scene.getEmissiveTriangles()[random.getUint32(0, scene.getEmissiveTrianglesCount() - 1)];
	return triangle.getRandomIntersection(scene, random);
}

CUDA_CALLABLE bool Integrator::isIntersectionVisible(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection)
{
	Vector3 originToEmissive = emissiveIntersection.position - origin.position;
	float distance = originToEmissive.length();
	Vector3 direction = originToEmissive / distance;

	Ray visibilityRay;
	visibilityRay.origin = origin.position;
	visibilityRay.direction = direction;
	visibilityRay.minDistance = scene.general.rayMinDistance;
	visibilityRay.maxDistance = distance - scene.general.rayMinDistance;
	visibilityRay.isVisibilityRay = true;
	visibilityRay.precalculate();

	Intersection visibilityIntersection;
	return !scene.intersect(visibilityRay, visibilityIntersection);
}

CUDA_CALLABLE DirectLightSample Integrator::calculateDirectLightSample(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection)
{
	Vector3 originToEmissive = emissiveIntersection.position - origin.position;
	float distance2 = originToEmissive.lengthSquared();
	float distance = std::sqrt(distance2);

	DirectLightSample result;
	result.direction = originToEmissive / distance;
	result.distance2 = distance2;
	result.originCosine = result.direction.dot(origin.normal);
	result.lightCosine = result.direction.dot(-emissiveIntersection.normal);

	if (result.originCosine <= 0.0f || result.lightCosine <= 0.0f)
	{
		result.visible = false;
		return result;
	}

	const Material& emissiveMaterial = scene.getMaterial(emissiveIntersection.materialIndex);

	result.emittance = emissiveMaterial.getEmittance(scene, emissiveIntersection.texcoord, emissiveIntersection.position);
	result.lightPdf = (1.0f / scene.getEmissiveTrianglesCount()) * (1.0f / emissiveIntersection.area) * (distance2 / result.lightCosine);
	result.visible = true;

	return result;
}

CUDA_CALLABLE float Integrator::balanceHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf)
{
	return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

CUDA_CALLABLE float Integrator::powerHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf)
{
	float f = nf * fPdf;
	float g = ng * gPdf;

	return (f * f) / (f * f + g * g);
}

VolumeEffect Integrator::calculateVolumeEffect(const Scene& scene, const Vector3& start, const Vector3& end, Random& random)
{
	Vector3 startToEnd = end - start;
	float distance = startToEnd.length();
	Vector3 direction = startToEnd / distance;
	float travelled = 0.0f;
	Color thickness(0.0f, 0.0f, 0.0f);
	Color emittance(0.0f, 0.0f, 0.0f);
	Color inscatter(0.0f, 0.0f, 0.0f);

	while (travelled < distance)
	{
		float density = 1.0f;
		float stepSize = scene.volume.stepSize;
		Vector3 position = start + travelled * direction;

		if (!scene.volume.constant)
		{
			density = scene.volume.noiseDensity.getNoise(position * scene.volume.noiseScale);
			stepSize = scene.volume.stepSize + random.getFloat() * scene.volume.stepSize;
		}

		travelled += stepSize;
		thickness += scene.volume.attenuationColor * density * stepSize * scene.volume.attenuationFactor;
		emittance += scene.volume.emissionColor * density * stepSize * scene.volume.emissionFactor;
		
		if (scene.volume.inscatter)
		{
			Intersection origin;
			Intersection emissiveIntersection = getRandomEmissiveIntersection(scene, random);

			origin.position = position;
			origin.normal = (emissiveIntersection.position - position).normalized();

			if (isIntersectionVisible(scene, origin, emissiveIntersection))
			{
				DirectLightSample lightSample = calculateDirectLightSample(scene, origin, emissiveIntersection);

				if (lightSample.visible)
					inscatter += scene.volume.inscatterColor * density * stepSize * scene.volume.inscatterFactor * lightSample.emittance * lightSample.lightCosine / lightSample.distance2;
			}
		}
	}

	VolumeEffect effect;
	effect.transmittance = scene.volume.attenuation ? Color::exp(-thickness) : Color(1.0f, 1.0f, 1.0f);
	effect.emittance = scene.volume.emission ? emittance : Color(0.0f, 0.0f, 0.0f);

	if (scene.volume.inscatter)
		effect.emittance += inscatter;

	return effect;
}

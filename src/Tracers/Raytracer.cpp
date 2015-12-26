// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/Raytracer.h"
#include "Scenes/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Tracing/Material.h"
#include "Tracing/Lights.h"
#include "Textures/Texture.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Tracing/ONB.h"

using namespace Raycer;

Color Raytracer::trace(const Scene& scene, const Ray& ray, std::mt19937& generator)
{
	Intersection intersection;
	return traceRecursive(scene, ray, intersection, 0, generator);
}

Color Raytracer::traceRecursive(const Scene& scene, const Ray& ray, Intersection& intersection, uint64_t iteration, std::mt19937& generator)
{
	Color finalColor = scene.general.backgroundColor;
	
	scene.intersect(ray, intersection);

	if (!intersection.wasFound)
		return finalColor;

	const Material* material = intersection.material;

	if (material->skipLighting)
	{
		finalColor = material->diffuseReflectance;

		if (material->diffuseMapTexture != nullptr)
			finalColor = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position) * material->diffuseMapTexture->intensity;

		if (scene.simpleFog.enabled)
		{
			if (ray.direction.dot(intersection.normal) < 0.0) // is outside
				finalColor = calculateSimpleFogColor(scene, intersection, finalColor);
		}
		
		return finalColor;
	}

	if (scene.general.enableNormalMapping && material->normalMapTexture != nullptr)
		calculateNormalMapping(intersection);

	double rayReflectance, rayTransmittance;
	Color reflectedColor, transmittedColor;

	calculateRayReflectanceAndTransmittance(ray, intersection, rayReflectance, rayTransmittance);

	if (rayReflectance > 0.0 && iteration < scene.general.maxRayIterations)
		reflectedColor = calculateReflectedColor(scene, ray, intersection, rayReflectance, iteration, generator);

	if (rayTransmittance > 0.0 && iteration < scene.general.maxRayIterations)
		transmittedColor = calculateTransmittedColor(scene, ray, intersection, rayTransmittance, iteration, generator);

	Color lightColor = calculateLightColor(scene, ray, intersection, generator);

	finalColor = lightColor + reflectedColor + transmittedColor;

	if (ray.direction.dot(intersection.normal) < 0.0) // is outside
	{
		if (scene.simpleFog.enabled)
			finalColor = calculateSimpleFogColor(scene, intersection, finalColor);
	}
	
	return finalColor;
}

void Raytracer::calculateNormalMapping(Intersection& intersection)
{
	Color normalColor = intersection.material->normalMapTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0 - 1.0, normalColor.g * 2.0 - 1.0, normalColor.b);
	ONB& onb = intersection.onb;
	Vector3 mappedNormal = onb.u * normal.x + onb.v * normal.y + onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

void Raytracer::calculateRayReflectanceAndTransmittance(const Ray& ray, const Intersection& intersection, double& rayReflectance, double& rayTransmittance)
{
	const Material* material = intersection.material;

	double fresnelReflectance = 1.0;
	double fresnelTransmittance = 1.0;

	if (material->fresnelReflection)
	{
		double cosine = ray.direction.dot(intersection.normal);
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

Color Raytracer::calculateReflectedColor(const Scene& scene, const Ray& ray, const Intersection& intersection, double rayReflectance, uint64_t iteration, std::mt19937& generator)
{
	const Material* material = intersection.material;

	Vector3 reflectionDirection = ray.direction + 2.0 * -ray.direction.dot(intersection.normal) * intersection.normal;
	reflectionDirection.normalize();

	Color reflectedColor;
	bool isOutside = ray.direction.dot(intersection.normal) < 0.0;

	Ray reflectedRay;
	Intersection reflectedIntersection;

	reflectedRay.origin = intersection.position + reflectionDirection * scene.general.rayStartOffset;
	reflectedRay.direction = reflectionDirection;
	reflectedRay.precalculate();

	reflectedColor = traceRecursive(scene, reflectedRay, reflectedIntersection, iteration + 1, generator) * rayReflectance;

	// only attenuate if ray has traveled inside a primitive
	if (!isOutside && reflectedIntersection.wasFound && material->attenuating)
	{
		double a = exp(-material->attenuationFactor * reflectedIntersection.distance);
		reflectedColor = Color::lerp(material->attenuationColor, reflectedColor, a);
	}

	return reflectedColor;
}

Color Raytracer::calculateTransmittedColor(const Scene& scene, const Ray& ray, const Intersection& intersection, double rayTransmittance, uint64_t iteration, std::mt19937& generator)
{
	const Material* material = intersection.material;

	double cosine1 = ray.direction.dot(intersection.normal);
	bool isOutside = cosine1 < 0.0;
	double n1 = isOutside ? 1.0 : material->refractiveIndex;
	double n2 = isOutside ? material->refractiveIndex : 1.0;
	double n3 = n1 / n2;
	double cosine2 = 1.0 - (n3 * n3) * (1.0 - cosine1 * cosine1);

	Color transmittedColor;

	// total internal reflection -> no transmission
	if (cosine2 <= 0.0)
		return transmittedColor;

	Vector3 transmissionDirection = ray.direction * n3 + (std::abs(cosine1) * n3 - sqrt(cosine2)) * intersection.normal;
	transmissionDirection.normalize();

	Ray transmittedRay;
	Intersection transmittedIntersection;

	transmittedRay.origin = intersection.position + transmissionDirection * scene.general.rayStartOffset;
	transmittedRay.direction = transmissionDirection;
	transmittedRay.precalculate();

	transmittedColor = traceRecursive(scene, transmittedRay, transmittedIntersection, iteration + 1, generator) * rayTransmittance;

	// only attenuate if ray has traveled inside a primitive
	if (isOutside && transmittedIntersection.wasFound && material->attenuating)
	{
		double a = exp(-material->attenuationFactor * transmittedIntersection.distance);
		transmittedColor = Color::lerp(material->attenuationColor, transmittedColor, a);
	}

	return transmittedColor;
}

Color Raytracer::calculateLightColor(const Scene& scene, const Ray& ray, const Intersection& intersection, std::mt19937& generator)
{
	Color lightColor;
	Vector3 directionToCamera = -ray.direction;
	const Material* material = intersection.material;

	Color mappedAmbientReflectance = Color(1.0, 1.0, 1.0);
	Color mappedDiffuseReflectance = Color(1.0, 1.0, 1.0);
	Color mappedSpecularReflectance = Color(1.0, 1.0, 1.0);

	if (material->ambientMapTexture != nullptr)
		mappedAmbientReflectance = material->ambientMapTexture->getColor(intersection.texcoord, intersection.position) * material->ambientMapTexture->intensity;

	if (material->diffuseMapTexture != nullptr)
		mappedDiffuseReflectance = material->diffuseMapTexture->getColor(intersection.texcoord, intersection.position) * material->diffuseMapTexture->intensity;

	if (material->specularMapTexture != nullptr)
		mappedSpecularReflectance = material->specularMapTexture->getColor(intersection.texcoord, intersection.position) * material->specularMapTexture->intensity;

	Color finalAmbientReflectance = material->ambientReflectance * mappedAmbientReflectance;
	Color finalDiffuseReflectance = material->diffuseReflectance * mappedDiffuseReflectance;
	Color finalSpecularReflectance = material->specularReflectance * mappedSpecularReflectance;
	
	lightColor += scene.lights.ambientLight.color * scene.lights.ambientLight.intensity * finalAmbientReflectance;

	for (const DirectionalLight& light : scene.lights.directionalLights)
	{
		Color directionalLightColor = calculatePhongShadingColor(intersection.normal, -light.direction, directionToCamera, light, finalDiffuseReflectance, finalSpecularReflectance, material->shininess);
		double shadowAmount = calculateShadowAmount(scene, ray, intersection, light);
		lightColor += directionalLightColor * (1.0 - shadowAmount);
	}

	for (const PointLight& light : scene.lights.pointLights)
	{
		Vector3 directionToLight = (light.position - intersection.position);
		double distanceToLight = directionToLight.length();
		directionToLight.normalize();

		Color pointLightColor = calculatePhongShadingColor(intersection.normal, directionToLight, directionToCamera, light, finalDiffuseReflectance, finalSpecularReflectance, material->shininess);
		double shadowAmount = calculateShadowAmount(scene, ray, intersection, light, generator);
		double distanceAttenuation = std::min(1.0, distanceToLight / light.maxDistance);
		distanceAttenuation = 1.0 - pow(distanceAttenuation, light.attenuation);

		lightColor += pointLightColor * distanceAttenuation * (1.0 - shadowAmount);
	}

	return lightColor;
}

Color Raytracer::calculatePhongShadingColor(const Vector3& normal, const Vector3& directionToLight, const Vector3& directionToCamera, const Light& light, const Color& diffuseReflectance, const Color& specularReflectance, double shininess)
{
	Color phongColor;

	double diffuseAmount = directionToLight.dot(normal);

	if (diffuseAmount > 0.0)
	{
		phongColor = light.color * light.intensity * diffuseAmount * diffuseReflectance;

		if (!specularReflectance.isZero())
		{
			Vector3 reflectionDirection = ((2.0 * diffuseAmount * normal) - directionToLight).normalized();
			double specularAmount = reflectionDirection.dot(directionToCamera);

			if (specularAmount > 0.0)
				phongColor += light.color * light.intensity * pow(specularAmount, shininess) * specularReflectance;
		}
	}

	return phongColor;
}

Color Raytracer::calculateSimpleFogColor(const Scene& scene, const Intersection& intersection, const Color& pixelColor)
{
	double t1 = intersection.distance / scene.simpleFog.distance;
	t1 = std::max(0.0, std::min(t1, 1.0));
	t1 = pow(t1, scene.simpleFog.steepness);

	if (scene.simpleFog.heightDispersion && intersection.position.y > 0.0)
	{
		double t2 = intersection.position.y / scene.simpleFog.height;
		t2 = std::max(0.0, std::min(t2, 1.0));
		t2 = pow(t2, scene.simpleFog.heightSteepness);
		t2 = 1.0 - t2;
		t1 *= t2;
	}

	return Color::lerp(pixelColor, scene.simpleFog.color, t1);
}

double Raytracer::calculateShadowAmount(const Scene& scene, const Ray& ray, const Intersection& intersection, const DirectionalLight& light)
{
	Vector3 directionToLight = -light.direction;

	Ray shadowRay;
	Intersection shadowIntersection;

	shadowRay.origin = intersection.position + directionToLight * scene.general.rayStartOffset;
	shadowRay.direction = directionToLight;
	shadowRay.isShadowRay = true;
	shadowRay.fastOcclusion = true;
	shadowRay.maxDistance = std::numeric_limits<double>::max();
	shadowRay.time = ray.time;
	shadowRay.precalculate();

	if (scene.intersect(shadowRay, shadowIntersection))
		return 1.0;

	return 0.0;
}

double Raytracer::calculateShadowAmount(const Scene& scene, const Ray& ray, const Intersection& intersection, const PointLight& light, std::mt19937& generator)
{
	Vector3 directionToLight = (light.position - intersection.position).normalized();

	if (!light.enableAreaLight)
	{
		Ray shadowRay;
		Intersection shadowIntersection;

		shadowRay.origin = intersection.position + directionToLight * scene.general.rayStartOffset;
		shadowRay.direction = directionToLight;
		shadowRay.isShadowRay = true;
		shadowRay.fastOcclusion = true;
		shadowRay.maxDistance = (light.position - intersection.position).length();
		shadowRay.time = ray.time;
		shadowRay.precalculate();

		if (scene.intersect(shadowRay, shadowIntersection))
			return 1.0;

		return 0.0;
	}

	Vector3 lightRight = directionToLight.cross(Vector3::ALMOST_UP).normalized();
	Vector3 lightUp = lightRight.cross(directionToLight).normalized();

	Sampler* sampler = samplers[light.areaLightSamplerType].get();
	std::uniform_int_distribution<uint64_t> randomPermutation;
	uint64_t permutation = randomPermutation(generator);

	double shadowAmount = 0.0;
	uint64_t n = light.areaLightSampleCountSqrt;

	for (uint64_t y = 0; y < n; ++y)
	{
		for (uint64_t x = 0; x < n; ++x)
		{
			Vector2 jitter = sampler->getDiscSample(x, y, n, n, permutation, generator) * light.areaLightRadius;
			Vector3 newLightPosition = light.position + jitter.x * lightRight + jitter.y * lightUp;
			Vector3 newDirectionToLight = (newLightPosition - intersection.position).normalized();

			Ray shadowRay;
			Intersection shadowIntersection;

			shadowRay.origin = intersection.position + newDirectionToLight * scene.general.rayStartOffset;
			shadowRay.direction = newDirectionToLight;
			shadowRay.isShadowRay = true;
			shadowRay.fastOcclusion = true;
			shadowRay.maxDistance = (newLightPosition - intersection.position).length();
			shadowRay.time = ray.time;
			shadowRay.precalculate();

			if (scene.intersect(shadowRay, shadowIntersection))
				shadowAmount += 1.0;
		}
	}

	return shadowAmount / (double(n) * double(n));
}

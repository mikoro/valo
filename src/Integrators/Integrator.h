// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Integrators/PathIntegrator.h"
#include "Integrators/DotIntegrator.h"
#include "Integrators/AmbientOcclusionIntegrator.h"
#include "Integrators/DirectLightIntegrator.h"
#include "Materials/Material.h"

namespace Valo
{
	class Color;
	class Scene;
	class Ray;
	class Random;

	enum class IntegratorType { PATH, DOT, AMBIENT_OCCLUSION, DIRECT_LIGHT };

	struct DirectLightSample
	{
		Color emittance;
		Vector3 direction;
		float distance2 = 0.0f;
		float originCosine = 0.0f;
		float lightCosine = 0.0f;
		float lightPdf = 0.0f;
		bool visible = false;
	};

	struct VolumeEffect
	{
		Color transmittance;
		Color emittance;
	};

	class Integrator
	{
	public:

		CUDA_CALLABLE Color calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;

		std::string getName() const;

		CUDA_CALLABLE static Intersection getRandomEmissiveIntersection(const Scene& scene, Random& random);
		CUDA_CALLABLE static bool isIntersectionVisible(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection);
		CUDA_CALLABLE static DirectLightSample calculateDirectLightSample(const Scene& scene, const Intersection& origin, const Intersection& emissiveIntersection);
		CUDA_CALLABLE static float balanceHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf);
		CUDA_CALLABLE static float powerHeuristic(uint32_t nf, float fPdf, uint32_t ng, float gPdf);
		CUDA_CALLABLE static VolumeEffect calculateVolumeEffect(const Scene& scene, const Vector3& start, const Vector3& end, Random& random);

		IntegratorType type = IntegratorType::PATH;

		PathIntegrator pathIntegrator;
		DotIntegrator dotIntegrator;
		AmbientOcclusionIntegrator aoIntegrator;
		DirectLightIntegrator directIntegrator;
	};
}

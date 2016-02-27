// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Rendering/Color.h"
#include "Math/Vector2.h"

namespace Raycer
{
	class Scene;
	class Intersection;
	class Light;
	class Random;
	class RandomSampler;
	class Vector3;
	class Texture;

	enum class MaterialType { DEFAULT };

	class Material
	{
	public:

		virtual ~Material() {}

		virtual Color getColor(const Scene& scene, const Intersection& intersection, const Light& light, Random& random) = 0;

		virtual Vector3 getDirection(const Intersection& intersection, RandomSampler& sampler, Random& random) = 0;
		virtual float getProbability(const Intersection& intersection, const Vector3& out) = 0;
		virtual Color getBrdf(const Intersection& intersection, const Vector3& out) = 0;

		virtual Color getReflectance(const Intersection& intersection);
		virtual Color getEmittance(const Intersection& intersection);
		virtual Color getAmbientReflectance(const Intersection& intersection);
		virtual Color getDiffuseReflectance(const Intersection& intersection);
		virtual Color getSpecularReflectance(const Intersection& intersection);

		virtual bool isEmissive();

		uint64_t id = 0;

		bool skipLighting = false;
		bool nonShadowing = false;
		bool normalInterpolation = true;
		bool autoInvertNormal = true;
		bool invertNormal = false;
		bool fresnelReflection = false;
		bool attenuating = false;
		float shininess = 2.0f;
		float refractiveIndex = 1.0f;
		float rayReflectance = 0.0f;
		float rayTransmittance = 0.0f;
		float attenuationFactor = 1.0f;
		Color attenuationColor = Color(0.0f, 0.0f, 0.0f);
		Vector2 texcoordScale = Vector2(1.0f, 1.0f);

		Color reflectance = Color(0.0f, 0.0f, 0.0f);
		uint64_t reflectanceMapTextureId = 0;
		Texture* reflectanceMapTexture = nullptr;

		Color emittance = Color(0.0f, 0.0f, 0.0f);
		uint64_t emittanceMapTextureId = 0;
		Texture* emittanceMapTexture = nullptr;

		Color ambientReflectance = Color(0.0f, 0.0f, 0.0f);
		uint64_t ambientMapTextureId = 0;
		Texture* ambientMapTexture = nullptr;

		Color diffuseReflectance = Color(0.0f, 0.0f, 0.0f);
		uint64_t diffuseMapTextureId = 0;
		Texture* diffuseMapTexture = nullptr;

		Color specularReflectance = Color(0.0f, 0.0f, 0.0f);
		uint64_t specularMapTextureId = 0;
		Texture* specularMapTexture = nullptr;

		uint64_t normalMapTextureId = 0;
		Texture* normalMapTexture = nullptr;

		uint64_t maskMapTextureId = 0;
		Texture* maskMapTexture = nullptr;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(skipLighting),
				CEREAL_NVP(nonShadowing),
				CEREAL_NVP(normalInterpolation),
				CEREAL_NVP(autoInvertNormal),
				CEREAL_NVP(invertNormal),
				CEREAL_NVP(fresnelReflection),
				CEREAL_NVP(attenuating),
				CEREAL_NVP(shininess),
				CEREAL_NVP(refractiveIndex),
				CEREAL_NVP(rayReflectance),
				CEREAL_NVP(rayTransmittance),
				CEREAL_NVP(attenuationFactor),
				CEREAL_NVP(attenuationColor),
				CEREAL_NVP(texcoordScale),
				CEREAL_NVP(reflectance),
				CEREAL_NVP(reflectanceMapTextureId),
				CEREAL_NVP(emittance),
				CEREAL_NVP(emittanceMapTextureId),
				CEREAL_NVP(ambientReflectance),
				CEREAL_NVP(ambientMapTextureId),
				CEREAL_NVP(diffuseReflectance),
				CEREAL_NVP(diffuseMapTextureId),
				CEREAL_NVP(specularReflectance),
				CEREAL_NVP(specularMapTextureId),
				CEREAL_NVP(normalMapTextureId),
				CEREAL_NVP(maskMapTextureId));
		}
	};
}

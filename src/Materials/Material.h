// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Color.h"
#include "Math/Vector2.h"
#include "Materials/DiffuseMaterial.h"
#include "Materials/BlinnPhongMaterial.h"

namespace Raycer
{
	class Texture;

	enum class MaterialType { DIFFUSE, BLINN_PHONG };

	class Material
	{
	public:

		Vector3 getDirection(const Intersection& intersection, Random& random);
		Color getBrdf(const Intersection& intersection, const Vector3& out);
		float getPdf(const Intersection& intersection, const Vector3& out);

		bool isEmissive() const;
		Color getEmittance(const Vector2& texcoord, const Vector3& position) const;
		Color getReflectance(const Vector2& texcoord, const Vector3& position) const;

		uint32_t id = 0;
		MaterialType type = MaterialType::DIFFUSE;

		bool nonShadowing = false;
		bool normalInterpolation = true;
		bool autoInvertNormal = true;
		bool invertNormal = false;

		Vector2 texcoordScale = Vector2(1.0f, 1.0f);

		Color emittance = Color(0.0f, 0.0f, 0.0f);
		uint32_t emittanceTextureId = 0;
		Texture* emittanceTexture = nullptr;

		Color reflectance = Color(0.0f, 0.0f, 0.0f);
		uint32_t reflectanceTextureId = 0;
		Texture* reflectanceTexture = nullptr;

		uint32_t normalTextureId = 0;
		Texture* normalTexture = nullptr;

		uint32_t maskTextureId = 0;
		Texture* maskTexture = nullptr;

		DiffuseMaterial diffuseMaterial;
		BlinnPhongMaterial blinnPhongMaterial;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(type),
				CEREAL_NVP(nonShadowing),
				CEREAL_NVP(normalInterpolation),
				CEREAL_NVP(autoInvertNormal),
				CEREAL_NVP(invertNormal),
				CEREAL_NVP(texcoordScale),
				CEREAL_NVP(emittance),
				CEREAL_NVP(emittanceTextureId),
				CEREAL_NVP(reflectance),
				CEREAL_NVP(reflectanceTextureId),
				CEREAL_NVP(normalTextureId),
				CEREAL_NVP(maskTextureId),
				CEREAL_NVP(blinnPhongMaterial));
		}
	};
}

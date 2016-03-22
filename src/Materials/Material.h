// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
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

		CUDA_CALLABLE Vector3 getDirection(const Intersection& intersection, Random& random);
		CUDA_CALLABLE Color getBrdf(const Intersection& intersection, const Vector3& in, const Vector3& out);
		CUDA_CALLABLE float getPdf(const Intersection& intersection, const Vector3& out);

		CUDA_CALLABLE bool isEmissive() const;
		CUDA_CALLABLE Color getEmittance(const Vector2& texcoord, const Vector3& position) const;
		CUDA_CALLABLE Color getReflectance(const Vector2& texcoord, const Vector3& position) const;

		uint32_t id = 0;
		MaterialType type = MaterialType::DIFFUSE;

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
	};
}

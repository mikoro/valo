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
	enum class MaterialType { DIFFUSE, BLINN_PHONG };

	class Scene;

	class Material
	{
	public:

		CUDA_CALLABLE Vector3 getDirection(const Intersection& intersection, Random& random) const;
		CUDA_CALLABLE Color getBrdf(const Scene& scene, const Intersection& intersection, const Vector3& in, const Vector3& out) const;
		CUDA_CALLABLE float getPdf(const Intersection& intersection, const Vector3& out) const;

		CUDA_CALLABLE bool isEmissive() const;
		CUDA_CALLABLE Color getEmittance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
		CUDA_CALLABLE Color getReflectance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;

		uint32_t id = 0;
		MaterialType type = MaterialType::DIFFUSE;

		bool normalInterpolation = true;
		bool autoInvertNormal = true;
		bool invertNormal = false;

		Vector2 texcoordScale = Vector2(1.0f, 1.0f);

		Color emittance = Color(0.0f, 0.0f, 0.0f);
		uint32_t emittanceTextureId = 0;
		int32_t emittanceTextureIndex = -1;

		Color reflectance = Color(0.0f, 0.0f, 0.0f);
		uint32_t reflectanceTextureId = 0;
		int32_t reflectanceTextureIndex = -1;

		uint32_t normalTextureId = 0;
		int32_t normalTextureIndex = -1;

		uint32_t maskTextureId = 0;
		int32_t maskTextureIndex = -1;

		DiffuseMaterial diffuseMaterial;
		BlinnPhongMaterial blinnPhongMaterial;
	};
}

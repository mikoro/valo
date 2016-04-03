// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Scene;
	class Vector3;
	class Intersection;
	class Random;
	class Color;
	class Material;
	class Texture;

	class BlinnPhongMaterial
	{
	public:

		CUDA_CALLABLE Vector3 getDirection(const Material& material, const Intersection& intersection, Random& random) const;
		CUDA_CALLABLE Color getBrdf(const Scene& scene, const Material& material, const Intersection& intersection, const Vector3& in, const Vector3& out) const;
		CUDA_CALLABLE float getPdf(const Material& material, const Intersection& intersection, const Vector3& out) const;

		CUDA_CALLABLE Color getSpecularReflectance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;
		CUDA_CALLABLE Color getGlossiness(const Scene& scene, const Vector2& texcoord, const Vector3& position) const;

		Color specularReflectance = Color(0.0f, 0.0f, 0.0f);
		uint32_t specularReflectanceTextureId = 0;
		int32_t specularReflectanceTextureIndex = -1;

		Color glossiness = Color(1.0f, 1.0f, 1.0f);
		uint32_t glossinessTextureId = 0;
		int32_t glossinessTextureIndex = -1;
	};
}

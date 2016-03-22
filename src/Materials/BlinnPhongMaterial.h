// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Vector3;
	class Intersection;
	class Random;
	class Color;
	class Material;
	class Texture;

	class BlinnPhongMaterial
	{
	public:

		CUDA_CALLABLE Vector3 getDirection(const Material& material, const Intersection& intersection, Random& random);
		CUDA_CALLABLE Color getBrdf(const Material& material, const Intersection& intersection, const Vector3& in, const Vector3& out);
		CUDA_CALLABLE float getPdf(const Material& material, const Intersection& intersection, const Vector3& out);

		CUDA_CALLABLE Color getSpecularReflectance(const Vector2& texcoord, const Vector3& position) const;
		CUDA_CALLABLE Color getGlossiness(const Vector2& texcoord, const Vector3& position) const;

		Color specularReflectance = Color(0.0f, 0.0f, 0.0f);
		uint32_t specularReflectanceTextureId = 0;
		Texture* specularReflectanceTexture = nullptr;

		Color glossiness = Color(1.0f, 1.0f, 1.0f);
		uint32_t glossinessTextureId = 0;
		Texture* glossinessTexture = nullptr;
	};
}

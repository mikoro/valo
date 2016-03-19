// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

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

		Vector3 getDirection(const Material& material, const Intersection& intersection, Random& random);
		Color getBrdf(const Material& material, const Intersection& intersection, const Vector3& out);
		float getPdf(const Material& material, const Intersection& intersection, const Vector3& out);

		Color getSpecularReflectance(const Vector2& texcoord, const Vector3& position) const;
		Color getGlossiness(const Vector2& texcoord, const Vector3& position) const;

		Color specularReflectance = Color(0.0f, 0.0f, 0.0f);
		uint64_t specularReflectanceTextureId = 0;
		Texture* specularReflectanceTexture = nullptr;

		Color glossiness = Color(1.0f, 1.0f, 1.0f);
		uint64_t glossinessTextureId = 0;
		Texture* glossinessTexture = nullptr;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(specularReflectance),
				CEREAL_NVP(specularReflectanceTextureId),
				CEREAL_NVP(glossiness),
				CEREAL_NVP(glossinessTextureId));
		}
	};
}

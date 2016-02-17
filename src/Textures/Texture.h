// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Vector2;
	class Vector3;

	class Texture
	{
	public:

		virtual ~Texture() {}

		virtual void initialize(Scene& scene) = 0;

		virtual Color getColor(const Vector2& texcoord, const Vector3& position) const = 0;
		virtual float getValue(const Vector2& texcoord, const Vector3& position) const = 0;

		uint64_t id = 0;
		float intensity = 1.0f;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(intensity));
		}
	};
}

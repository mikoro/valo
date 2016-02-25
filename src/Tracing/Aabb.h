// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector3.h"

namespace Raycer
{
	class Ray;
	class EulerAngle;

	class Aabb
	{
	public:

		Aabb();

		static Aabb createFromMinMax(const Vector3& min, const Vector3& max);
		static Aabb createFromCenterExtent(const Vector3& center, const Vector3& extent);
		static Aabb createFromVertices(const Vector3& v0, const Vector3& v1, const Vector3& v2);

		static std::array<uint32_t, 4> intersects(const float* __restrict aabbMinX, const float* __restrict aabbMinY, const float* __restrict aabbMinZ, const float* __restrict aabbMaxX, const float* __restrict aabbMaxY, const float* __restrict aabbMaxZ, const Ray& ray);

		bool intersects(const Ray& ray) const;
		void expand(const Aabb& other);
		
		Vector3 getCenter() const;
		Vector3 getExtent() const;
		float getSurfaceArea() const;

		Vector3 min;
		Vector3 max;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(min),
				CEREAL_NVP(max));
		}
	};
}

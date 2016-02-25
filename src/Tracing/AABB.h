// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Math/Vector3.h"

namespace Raycer
{
	class Ray;
	class EulerAngle;

	class AABB
	{
	public:

		AABB();

		static AABB createFromMinMax(const Vector3& min, const Vector3& max);
		static AABB createFromCenterExtent(const Vector3& center, const Vector3& extent);
		static AABB createFromVertices(const Vector3& v0, const Vector3& v1, const Vector3& v2);

		template <uint64_t N>
		static std::array<uint32_t, N> intersects(const float* __restrict aabbMinX, const float* __restrict aabbMinY, const float* __restrict aabbMinZ, const float* __restrict aabbMaxX, const float* __restrict aabbMaxY, const float* __restrict aabbMaxZ, const Ray& ray);

		bool intersects(const Ray& ray) const;
		void expand(const AABB& other);
		
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

#include "Tracing/AABB.inl"

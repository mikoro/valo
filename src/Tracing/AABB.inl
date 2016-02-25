// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

namespace Raycer
{
	template <uint64_t N>
	std::array<uint32_t, N> AABB::intersects(
		const float* __restrict aabbMinX,
		const float* __restrict aabbMinY,
		const float* __restrict aabbMinZ,
		const float* __restrict aabbMaxX,
		const float* __restrict aabbMaxY,
		const float* __restrict aabbMaxZ,
		const Ray& ray)
	{
		const float originX = ray.origin.x;
		const float originY = ray.origin.y;
		const float originZ = ray.origin.z;

		const float inverseDirectionX = ray.inverseDirection.x;
		const float inverseDirectionY = ray.inverseDirection.y;
		const float inverseDirectionZ = ray.inverseDirection.z;

		const float minDistance = ray.minDistance;
		const float maxDistance = ray.maxDistance;

		alignas(16) uint32_t result[N];

#if !defined(_MSC_VER) || defined(__INTEL_COMPILER)
		__assume_aligned(aabbMinX, 16);
		__assume_aligned(aabbMinY, 16);
		__assume_aligned(aabbMinZ, 16);
		__assume_aligned(aabbMaxX, 16);
		__assume_aligned(aabbMaxY, 16);
		__assume_aligned(aabbMaxZ, 16);
#endif

#ifdef __INTEL_COMPILER
#pragma vector always assert aligned
#endif
		for (uint32_t i = 0; i < N; ++i)
		{
			const float tx0 = (aabbMinX[i] - originX) * inverseDirectionX;
			const float tx1 = (aabbMaxX[i] - originX) * inverseDirectionX;

			float tmin = MIN(tx0, tx1);
			float tmax = MAX(tx0, tx1);

			const float ty0 = (aabbMinY[i] - originY) * inverseDirectionY;
			const float ty1 = (aabbMaxY[i] - originY) * inverseDirectionY;

			tmin = MAX(tmin, MIN(ty0, ty1));
			tmax = MIN(tmax, MAX(ty0, ty1));

			const float tz0 = (aabbMinZ[i] - originZ) * inverseDirectionZ;
			const float tz1 = (aabbMaxZ[i] - originZ) * inverseDirectionZ;

			tmin = MAX(tmin, MIN(tz0, tz1));
			tmax = MIN(tmax, MAX(tz0, tz1));

			result[i] = tmax >= MAX(tmin, 0.0f) && tmin < maxDistance && tmax > minDistance;
		}

		std::array<uint32_t, N> resultArray;
		memcpy(resultArray.data(), result, sizeof(resultArray));
		return resultArray;
	}
}

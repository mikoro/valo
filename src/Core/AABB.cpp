// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/AABB.h"
#include "Core/Ray.h"

using namespace Raycer;

AABB::AABB()
{
	min.x = min.y = min.z = std::numeric_limits<float>::max();
	max.x = max.y = max.z = std::numeric_limits<float>::lowest();
}

AABB AABB::createFromMinMax(const Vector3& min_, const Vector3& max_)
{
	AABB aabb;

	aabb.min = min_;
	aabb.max = max_;

	return aabb;
}

AABB AABB::createFromCenterExtent(const Vector3& center, const Vector3& extent)
{
	AABB aabb;

	aabb.min = center - extent / 2.0f;
	aabb.max = center + extent / 2.0f;

	return aabb;
}

AABB AABB::createFromVertices(const Vector3& v0, const Vector3& v1, const Vector3& v2)
{
	Vector3 min_;

	min_.x = std::min(v0.x, std::min(v1.x, v2.x));
	min_.y = std::min(v0.y, std::min(v1.y, v2.y));
	min_.z = std::min(v0.z, std::min(v1.z, v2.z));

	Vector3 max_;

	max_.x = std::max(v0.x, std::max(v1.x, v2.x));
	max_.y = std::max(v0.y, std::max(v1.y, v2.y));
	max_.z = std::max(v0.z, std::max(v1.z, v2.z));

	return AABB::createFromMinMax(min_, max_);
}

// http://tavianator.com/fast-branchless-raybounding-box-intersections-part-2-nans/
bool AABB::intersects(const Ray& ray) const
{
	float tmin = ((&min)[ray.directionIsNegative[0]].x - ray.origin.x) * ray.inverseDirection.x;
	float tmax = ((&min)[1 - ray.directionIsNegative[0]].x - ray.origin.x) * ray.inverseDirection.x;
	float tymin = ((&min)[ray.directionIsNegative[1]].y - ray.origin.y) * ray.inverseDirection.y;
	float tymax = ((&min)[1 - ray.directionIsNegative[1]].y - ray.origin.y) * ray.inverseDirection.y;

	if (tmin > tymax || tymin > tmax)
		return false;

	if (tymin > tmin)
		tmin = tymin;

	if (tymax < tmax)
		tmax = tymax;

	float tzmin = ((&min)[ray.directionIsNegative[2]].z - ray.origin.z) * ray.inverseDirection.z;
	float tzmax = ((&min)[1 - ray.directionIsNegative[2]].z - ray.origin.z) * ray.inverseDirection.z;

	if (tmin > tzmax || tzmin > tmax)
		return false;

	if (tzmin > tmin)
		tmin = tzmin;

	if (tzmax < tmax)
		tmax = tzmax;

	return (tmin < ray.maxDistance) && (tmax > ray.minDistance);
}

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

template <uint32_t N>
std::array<bool, N> AABB::intersects(
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

	std::array<bool, N> result;

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

	return result;
}

template std::array<bool, 4> AABB::intersects<4>(const float* __restrict aabbMinX, const float* __restrict aabbMinY, const float* __restrict aabbMinZ, const float* __restrict aabbMaxX, const float* __restrict aabbMaxY, const float* __restrict aabbMaxZ, const Ray& ray);
template std::array<bool, 8> AABB::intersects<8>(const float* __restrict aabbMinX, const float* __restrict aabbMinY, const float* __restrict aabbMinZ, const float* __restrict aabbMaxX, const float* __restrict aabbMaxY, const float* __restrict aabbMaxZ, const Ray& ray);
template std::array<bool, 16> AABB::intersects<16>(const float* __restrict aabbMinX, const float* __restrict aabbMinY, const float* __restrict aabbMinZ, const float* __restrict aabbMaxX, const float* __restrict aabbMaxY, const float* __restrict aabbMaxZ, const Ray& ray);

void AABB::expand(const AABB& other)
{
	if (other.min.x < min.x)
		min.x = other.min.x;

	if (other.min.y < min.y)
		min.y = other.min.y;

	if (other.min.z < min.z)
		min.z = other.min.z;

	if (other.max.x > max.x)
		max.x = other.max.x;

	if (other.max.y > max.y)
		max.y = other.max.y;

	if (other.max.z > max.z)
		max.z = other.max.z;
}

Vector3 AABB::getCenter() const
{
	return (min + max) * 0.5;
}

Vector3 AABB::getExtent() const
{
	return max - min;
}

float AABB::getSurfaceArea() const
{
	Vector3 extent = getExtent();
	return 2.0f * (extent.x * extent.y + extent.z * extent.y + extent.x * extent.z);
}

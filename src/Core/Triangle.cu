// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Common.h"
#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Core/Triangle.h"
#include "Materials/Material.h"
#include "Math/ONB.h"
#include "Textures/Texture.h"
#include "Math/Random.h"

using namespace Raycer;

void Triangle::initialize()
{
	Vector3 v0tov1 = vertices[1] - vertices[0];
	Vector3 v0tov2 = vertices[2] - vertices[0];
	Vector2 t0tot1 = texcoords[1] - texcoords[0];
	Vector2 t0tot2 = texcoords[2] - texcoords[0];

	Vector3 cross = v0tov1.cross(v0tov2);
	normal = cross.normalized();
	area = 0.5f * cross.length();

	float denominator = t0tot1.x * t0tot2.y - t0tot1.y * t0tot2.x;

	// tangent space aligned to texcoords
	if (std::abs(denominator) > 0.0000000001f)
	{
		float r = 1.0f / denominator;
		tangent = (v0tov1 * t0tot2.y - v0tov2 * t0tot1.y) * r;
		bitangent = (v0tov1 * t0tot2.x - v0tov2 * t0tot1.x) * r;
		tangent.normalize();
		bitangent.normalize();
	}
	else
	{
		tangent = normal.cross(Vector3::almostUp()).normalized();
		bitangent = tangent.cross(normal).normalized();
	}
}

// Möller-Trumbore algorithm
// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
CUDA_CALLABLE bool Triangle::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	if (ray.isVisibilityRay && intersection.wasFound)
		return true;

	Vector3 v0v1 = vertices[1] - vertices[0];
	Vector3 v0v2 = vertices[2] - vertices[0];

	Vector3 pvec = ray.direction.cross(v0v2);
	float determinant = v0v1.dot(pvec);

	// ray and triangle are parallel -> no intersection
	if (std::abs(determinant) < 0.0000000001f)
		return false;

	float invDeterminant = 1.0f / determinant;

	Vector3 tvec = ray.origin - vertices[0];
	float u = tvec.dot(pvec) * invDeterminant;

	if (u < 0.0f || u > 1.0f)
		return false;

	Vector3 qvec = tvec.cross(v0v1);
	float v = ray.direction.dot(qvec) * invDeterminant;

	if (v < 0.0f || (u + v) > 1.0f)
		return false;

	float distance = v0v2.dot(qvec) * invDeterminant;

	if (distance < 0.0f)
		return false;

	if (distance < ray.minDistance || distance > ray.maxDistance)
		return false;

	if (distance > intersection.distance)
		return false;

	return calculateIntersectionData(scene, ray, *this, intersection, distance, u, v);
}

template <uint32_t N>
CUDA_CALLABLE bool Triangle::intersect(
	const float* __restrict vertex1X,
	const float* __restrict vertex1Y,
	const float* __restrict vertex1Z,
	const float* __restrict vertex2X,
	const float* __restrict vertex2Y,
	const float* __restrict vertex2Z,
	const float* __restrict vertex3X,
	const float* __restrict vertex3Y,
	const float* __restrict vertex3Z,
	const uint32_t* __restrict triangleIndices,
	const Scene& scene,
	const Ray& ray,
	Intersection& intersection)
{
	if (ray.isVisibilityRay && intersection.wasFound)
		return true;

	const float originX = ray.origin.x;
	const float originY = ray.origin.y;
	const float originZ = ray.origin.z;

	const float directionX = ray.direction.x;
	const float directionY = ray.direction.y;
	const float directionZ = ray.direction.z;

	const float minDistance = ray.minDistance;
	const float maxDistance = ray.maxDistance;
	const float intersectionDistance = intersection.distance;

	ALIGN(16) uint32_t hits[N];
	ALIGN(16) float distances[N];
	ALIGN(16) float uValues[N];
	ALIGN(16) float vValues[N];

	memset(hits, 1, sizeof(hits));

	for (uint32_t i = 0; i < N; ++i)
	{
		const float v0v1X = vertex2X[i] - vertex1X[i];
		const float v0v1Y = vertex2Y[i] - vertex1Y[i];
		const float v0v1Z = vertex2Z[i] - vertex1Z[i];

		const float v0v2X = vertex3X[i] - vertex1X[i];
		const float v0v2Y = vertex3Y[i] - vertex1Y[i];
		const float v0v2Z = vertex3Z[i] - vertex1Z[i];

		// cross product
		const float pvecX = directionY * v0v2Z - directionZ * v0v2Y;
		const float pvecY = directionZ * v0v2X - directionX * v0v2Z;
		const float pvecZ = directionX * v0v2Y - directionY * v0v2X;

		// dot product
		const float determinant = v0v1X * pvecX + v0v1Y * pvecY + v0v1Z * pvecZ;

		if (std::abs(determinant) < 0.0000000001f)
			hits[i] = 0;

		const float invDeterminant = 1.0f / determinant;

		const float tvecX = originX - vertex1X[i];
		const float tvecY = originY - vertex1Y[i];
		const float tvecZ = originZ - vertex1Z[i];

		// dot product
		const float u = (tvecX * pvecX + tvecY * pvecY + tvecZ * pvecZ) * invDeterminant;

		if (u < 0.0f || u > 1.0f)
			hits[i] = 0;

		// cross product
		const float qvecX = tvecY * v0v1Z - tvecZ * v0v1Y;
		const float qvecY = tvecZ * v0v1X - tvecX * v0v1Z;
		const float qvecZ = tvecX * v0v1Y - tvecY * v0v1X;

		// dot product
		const float v = (directionX * qvecX + directionY * qvecY + directionZ * qvecZ) * invDeterminant;

		if (v < 0.0f || (u + v) > 1.0f)
			hits[i] = 0;

		const float t = (v0v2X * qvecX + v0v2Y * qvecY + v0v2Z * qvecZ) * invDeterminant;

		if (t < 0.0f)
			hits[i] = 0;

		if (t < minDistance || t > maxDistance)
			hits[i] = 0;

		if (t > intersectionDistance)
			hits[i] = 0;

		uValues[i] = u;
		vValues[i] = v;
		distances[i] = t;
	}

	float distance, u, v;
	uint32_t triangleIndex;

	// if this isn't in its own function, the icc vectorizer gets confused
	if (!findIntersectionValues<N>(hits, distances, uValues, vValues, triangleIndices, distance, u, v, triangleIndex))
		return false;

	return calculateIntersectionData(scene, ray, scene.trianglesPtr[triangleIndex], intersection, distance, u, v);
}

template <uint32_t N>
CUDA_CALLABLE bool Triangle::findIntersectionValues(const uint32_t* hits, const float* distances, const float* uValues, const float* vValues, const uint32_t* triangleIndices, float& distance, float& u, float& v, uint32_t& triangleIndex)
{
	bool wasFound = false;
	distance = FLT_MAX;

	for (uint32_t i = 0; i < N; ++i)
	{
		if (hits[i] && distances[i] < distance)
		{
			wasFound = true;

			u = uValues[i];
			v = vValues[i];
			distance = distances[i];
			triangleIndex = triangleIndices[i];
		}
	}

	return wasFound;
}

template bool Triangle::intersect<4>(const float* __restrict vertex1X, const float* __restrict vertex1Y, const float* __restrict vertex1Z, const float* __restrict vertex2X, const float* __restrict vertex2Y, const float* __restrict vertex2Z, const float* __restrict vertex3X, const float* __restrict vertex3Y, const float* __restrict vertex3Z, const uint32_t* __restrict triangleIndices, const Scene& scene, const Ray& ray, Intersection& intersection);
template bool Triangle::intersect<8>(const float* __restrict vertex1X, const float* __restrict vertex1Y, const float* __restrict vertex1Z, const float* __restrict vertex2X, const float* __restrict vertex2Y, const float* __restrict vertex2Z, const float* __restrict vertex3X, const float* __restrict vertex3Y, const float* __restrict vertex3Z, const uint32_t* __restrict triangleIndices, const Scene& scene, const Ray& ray, Intersection& intersection);
template bool Triangle::intersect<16>(const float* __restrict vertex1X, const float* __restrict vertex1Y, const float* __restrict vertex1Z, const float* __restrict vertex2X, const float* __restrict vertex2Y, const float* __restrict vertex2Z, const float* __restrict vertex3X, const float* __restrict vertex3Y, const float* __restrict vertex3Z, const uint32_t* __restrict triangleIndices, const Scene& scene, const Ray& ray, Intersection& intersection);

CUDA_CALLABLE bool Triangle::calculateIntersectionData(const Scene& scene, const Ray& ray, const Triangle& triangle, Intersection& intersection, float distance, float u, float v)
{
	float w = 1.0f - u - v;

	Vector3 intersectionPosition = ray.origin + (distance * ray.direction);
	Vector2 texcoord = (w * triangle.texcoords[0] + u * triangle.texcoords[1] + v * triangle.texcoords[2]) * triangle.material->texcoordScale;

	texcoord.x = texcoord.x - floor(texcoord.x);
	texcoord.y = texcoord.y - floor(texcoord.y);

	if (triangle.material->maskTexture != nullptr)
	{
		if (triangle.material->maskTexture->getColor(texcoord, intersectionPosition).r < 0.5f)
			return false;
	}

	Vector3 tempNormal = (scene.general.normalInterpolation && triangle.material->normalInterpolation) ? (w * triangle.normals[0] + u * triangle.normals[1] + v * triangle.normals[2]) : triangle.normal;

	if (triangle.material->invertNormal)
		tempNormal = -tempNormal;

	intersection.isBehind = ray.direction.dot(tempNormal) > 0.0f;

	if (triangle.material->autoInvertNormal && intersection.isBehind)
		tempNormal = -tempNormal;

	if (scene.general.interpolationVisualization)
	{
		intersection.color = w * Color::red() + u * Color::green() + v * Color::blue();
		intersection.hasColor = true;
	}

	intersection.wasFound = true;
	intersection.distance = distance;
	intersection.position = intersectionPosition;
	intersection.normal = tempNormal;
	intersection.texcoord = texcoord;
	intersection.onb = ONB(triangle.tangent, triangle.bitangent, tempNormal);
	intersection.material = triangle.material;

	return true;
}

CUDA_CALLABLE Intersection Triangle::getRandomIntersection(Random& random) const
{
	float r1 = random.getFloat();
	float r2 = random.getFloat();
	float sr1 = std::sqrt(r1);

	float u = 1.0f - sr1;
	float v = r2 * sr1;
	float w = 1.0f - u - v;

	Vector3 position = u * vertices[0] + v * vertices[1] + w * vertices[2];
	Vector3 tempNormal = material->normalInterpolation ? (w * normals[0] + u * normals[1] + v * normals[2]) : normal;
	Vector2 texcoord = (w * texcoords[0] + u * texcoords[1] + v * texcoords[2]) * material->texcoordScale;

	texcoord.x = texcoord.x - floor(texcoord.x);
	texcoord.y = texcoord.y - floor(texcoord.y);

	Intersection intersection;

	intersection.position = position;
	intersection.normal = tempNormal;
	intersection.texcoord = texcoord;
	intersection.onb = ONB(tangent, bitangent, tempNormal);
	intersection.material = material;

	return intersection;
}

AABB Triangle::getAABB() const
{
	return AABB::createFromVertices(vertices[0], vertices[1], vertices[2]);
}

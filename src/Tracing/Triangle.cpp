// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/Scene.h"
#include "Tracing/Triangle.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Tracing/ONB.h"
#include "Materials/Material.h"
#include "Textures/Texture.h"
#include "Utils/Random.h"

using namespace Raycer;

void Triangle::initialize()
{
	Vector3 v0tov1 = vertices[1] - vertices[0];
	Vector3 v0tov2 = vertices[2] - vertices[0];
	Vector2 t0tot1 = texcoords[1] - texcoords[0];
	Vector2 t0tot2 = texcoords[2] - texcoords[0];

	Vector3 cross = v0tov1.cross(v0tov2);
	normal = cross.normalized();

	float denominator = t0tot1.x * t0tot2.y - t0tot1.y * t0tot2.x;

	// tangent space aligned to texcoords
	if (std::abs(denominator) > std::numeric_limits<float>::epsilon())
	{
		float r = 1.0f / denominator;
		tangent = (v0tov1 * t0tot2.y - v0tov2 * t0tot1.y) * r;
		bitangent = (v0tov2 * t0tot1.x - v0tov1 * t0tot2.x) * r;
		tangent.normalize();
		bitangent.normalize();
	}
	else
	{
		tangent = normal.cross(Vector3::ALMOST_UP).normalized();
		bitangent = tangent.cross(normal).normalized();
	}
}

// Möller-Trumbore algorithm
// http://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
bool Triangle::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	if (ray.isShadowRay && material->nonShadowing)
		return false;

	if (ray.fastOcclusion && intersection.wasFound)
		return true;

	Vector3 v0v1 = vertices[1] - vertices[0];
	Vector3 v0v2 = vertices[2] - vertices[0];

	Vector3 pvec = ray.direction.cross(v0v2);
	float determinant = v0v1.dot(pvec);

	// ray and triangle are parallel -> no intersection
	if (std::abs(determinant) < std::numeric_limits<float>::epsilon())
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

	float t = v0v2.dot(qvec) * invDeterminant;

	if (t < 0.0f)
		return false;

	if (t < ray.minDistance || t > ray.maxDistance)
		return false;

	if (t > intersection.distance)
		return false;

	float w = 1.0f - u - v;

	Vector3 intersectionPosition = ray.origin + (t * ray.direction);
	Vector2 texcoord = (w * texcoords[0] + u * texcoords[1] + v * texcoords[2]) * material->texcoordScale;

	texcoord.x = texcoord.x - floor(texcoord.x);
	texcoord.y = texcoord.y - floor(texcoord.y);

	if (material->maskMapTexture != nullptr)
	{
		if (material->maskMapTexture->getValue(texcoord, intersectionPosition) < 0.5f)
			return false;
	}

	Vector3 tempNormal = (scene.general.normalInterpolation && material->normalInterpolation) ? (w * normals[0] + u * normals[1] + v * normals[2]) : normal;

	if (material->invertNormal)
		tempNormal = -tempNormal;

	intersection.isBehind = ray.direction.dot(tempNormal) > 0.0f;

	if (material->autoInvertNormal && intersection.isBehind)
		tempNormal = -tempNormal;

	if (scene.general.normalVisualization)
	{
		intersection.color.r = (tempNormal.x + 1.0f) / 2.0f;
		intersection.color.g = (tempNormal.y + 1.0f) / 2.0f;
		intersection.color.b = (tempNormal.z + 1.0f) / 2.0f;
		intersection.hasColor = true;
	}

	if (scene.general.interpolationVisualization)
	{
		intersection.color = w * Color::RED + u * Color::GREEN + v * Color::BLUE;
		intersection.hasColor = true;
	}

	intersection.wasFound = true;
	intersection.distance = t;
	intersection.position = intersectionPosition;
	intersection.normal = tempNormal;
	intersection.texcoord = texcoord;
	intersection.rayDirection = ray.direction;
	intersection.onb = ONB(tangent, bitangent, tempNormal);
	intersection.material = material;

	return true;
}

Intersection Triangle::getRandomIntersection(Random& random) const
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

float Triangle::getArea() const
{
	Vector3 v0tov1 = vertices[1] - vertices[0];
	Vector3 v0tov2 = vertices[2] - vertices[0];
	Vector3 cross = v0tov1.cross(v0tov2);

	return 0.5f * cross.length();
}

bool Triangle::intersect(
	const float* __restrict vertex1X,
	const float* __restrict vertex1Y,
	const float* __restrict vertex1Z,
	const float* __restrict vertex2X,
	const float* __restrict vertex2Y,
	const float* __restrict vertex2Z,
	const float* __restrict vertex3X,
	const float* __restrict vertex3Y,
	const float* __restrict vertex3Z,
	const uint32_t* __restrict triangleId,
	const Scene& scene,
	const Ray& ray,
	Intersection& intersection)
{
	const float originX = ray.origin.x;
	const float originY = ray.origin.y;
	const float originZ = ray.origin.z;

	const float directionX = ray.direction.x;
	const float directionY = ray.direction.y;
	const float directionZ = ray.direction.z;

	const float minDistance = ray.minDistance;
	const float maxDistance = ray.maxDistance;

	alignas(16) float result[4];
	memset(result, 1, sizeof(result));

#ifdef __INTEL_COMPILER
#pragma vector always assert aligned
#endif
	for (uint32_t i = 0; i < 4; ++i)
	{
		uint32_t wasFound = 1;

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
		
		wasFound = (std::abs(determinant) < 1.0f) ? 0 : 1;

		/*

		// ray and triangle are parallel -> no intersection
		if (std::abs(determinant) < std::numeric_limits<float>::epsilon())
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

		float t = v0v2.dot(qvec) * invDeterminant;

		if (t < 0.0f)
			return false;

		if (t < ray.minDistance || t > ray.maxDistance)
			return false;

		if (t > intersection.distance)
			return false;*/

		result[i] = wasFound;
	}

	return true;
}

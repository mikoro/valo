// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Core/Intersection.h"
#include "Core/Ray.h"
#include "Core/Scene.h"
#include "Core/Triangle.h"
#include "Materials/Material.h"
#include "Math/ONB.h"
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
	area = 0.5f * cross.length();

	float denominator = t0tot1.x * t0tot2.y - t0tot1.y * t0tot2.x;

	// tangent space aligned to texcoords
	if (std::abs(denominator) > std::numeric_limits<float>::epsilon())
	{
		float r = 1.0f / denominator;
		tangent = (v0tov1 * t0tot2.y - v0tov2 * t0tot1.y) * r;
		bitangent = (v0tov1 * t0tot2.x - v0tov2 * t0tot1.x) * r;
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
	if (ray.isVisibilityRay && intersection.wasFound)
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

	float distance = v0v2.dot(qvec) * invDeterminant;

	if (distance < 0.0f)
		return false;

	if (distance < ray.minDistance || distance > ray.maxDistance)
		return false;

	if (distance > intersection.distance)
		return false;

	return calculateIntersectionData(scene, ray, *this, intersection, distance, u, v);
}

bool Triangle::calculateIntersectionData(const Scene& scene, const Ray& ray, const Triangle& triangle, Intersection& intersection, float distance, float u, float v)
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
		intersection.color = w * Color::RED + u * Color::GREEN + v * Color::BLUE;
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
	return area;
}

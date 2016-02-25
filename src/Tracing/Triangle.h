// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Tracing/AABB.h"
#include "Math/Vector3.h"
#include "Math/Vector2.h"

namespace Raycer
{
	class Scene;
	class Ray;
	class Intersection;
	class Material;
	class Random;

	template <uint64_t N>
	struct TriangleSOA
	{
		std::array<float, N> vertex1X;
		std::array<float, N> vertex1Y;
		std::array<float, N> vertex1Z;
		std::array<float, N> vertex2X;
		std::array<float, N> vertex2Y;
		std::array<float, N> vertex2Z;
		std::array<float, N> vertex3X;
		std::array<float, N> vertex3Y;
		std::array<float, N> vertex3Z;
		std::array<uint32_t, N> triangleId;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(vertex1X),
				CEREAL_NVP(vertex1Y),
				CEREAL_NVP(vertex1Z),
				CEREAL_NVP(vertex2X),
				CEREAL_NVP(vertex2Y),
				CEREAL_NVP(vertex2Z),
				CEREAL_NVP(vertex3X),
				CEREAL_NVP(vertex3Y),
				CEREAL_NVP(vertex3Z),
				CEREAL_NVP(triangleId));
		}
	};

	using TriangleSOAVector4 = std::vector<TriangleSOA<4>, boost::alignment::aligned_allocator<TriangleSOA<4>, 16>>;
	using TriangleSOAVector8 = std::vector<TriangleSOA<8>, boost::alignment::aligned_allocator<TriangleSOA<8>, 16>>;

	class Triangle
	{
	public:

		void initialize();
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;
		Intersection getRandomIntersection(Random& random) const;
		AABB getAABB() const;
		float getArea() const;

		static bool intersect(const float* __restrict vertex1X, const float* __restrict vertex1Y, const float* __restrict vertex1Z, const float* __restrict vertex2X, const float* __restrict vertex2Y, const float* __restrict vertex2Z, const float* __restrict vertex3X, const float* __restrict vertex3Y, const float* __restrict vertex3Z, const uint32_t* __restrict triangleId, const Scene& scene, const Ray& ray, Intersection& intersection);

		uint64_t id = 0;
		uint64_t materialId = 0;

		Vector3 vertices[3];
		Vector3 normals[3];
		Vector2 texcoords[3];
		Vector3 normal;
		Vector3 tangent;
		Vector3 bitangent;
		
		Material* material = nullptr;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(materialId),
				CEREAL_NVP(vertices),
				CEREAL_NVP(normals),
				CEREAL_NVP(texcoords));
		}
	};
}

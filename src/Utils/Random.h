// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <limits>
#include <random>

namespace Raycer
{
	class Color;
	class Vector2;
	class Vector3;

	class Random
	{
	public:

		Random();
		Random(uint32_t seed);

		void initialize();
		void initialize(uint32_t seed);

		int32_t getInt32(int32_t min = std::numeric_limits<int32_t>::lowest(), int32_t max = std::numeric_limits<int32_t>::max());
		uint32_t getUint32(uint32_t min = std::numeric_limits<uint32_t>::lowest(), uint32_t max = std::numeric_limits<uint32_t>::max());
		float getFloat(float min = 0.0f, float max = 1.0f);
		double getDouble(double min = 0.0, double max = 1.0);

		Color getColor(bool randomAlpha = false);
		Vector2 getVector2();
		Vector3 getVector3();

		typedef uint32_t result_type;
		static result_type min() { return 0; }
		static result_type max() { return std::numeric_limits<uint32_t>::max(); }
		result_type operator()();

	private:

		std::mt19937 generator;
	};
}

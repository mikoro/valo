// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <limits>
#include <random>

#include "Rendering/Color.h"

namespace Raycer
{
	class Vector2;
	class Vector3;

	class Random
	{
	public:

		Random();
		Random(uint64_t seed);

		void initialize();
		void initialize(uint64_t seed);

		int64_t getInt64(int64_t min = std::numeric_limits<int64_t>::lowest(), int64_t max = std::numeric_limits<int64_t>::max());
		uint64_t getUint64(uint64_t min = std::numeric_limits<uint64_t>::lowest(), uint64_t max = std::numeric_limits<uint64_t>::max());
		float getFloat(float min = 0.0f, float max = 1.0f);
		double getDouble(double min = 0.0, double max = 1.0);

		Color getColor(bool randomAlpha = false);
		Vector2 getVector2();
		Vector3 getVector3();

		typedef uint64_t result_type;
		static result_type min();
		static result_type max();
		result_type operator()();

	private:

		std::mt19937_64 generator;
	};
}

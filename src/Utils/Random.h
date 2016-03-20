// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <limits>

namespace Raycer
{
	class Color;
	class Vector2;
	class Vector3;

	// http://www.pcg-random.org/
	class RandomGeneratorPCG
	{
	public:

		RandomGeneratorPCG();
		explicit RandomGeneratorPCG(uint64_t seed);

		void seed(uint64_t seed);

		typedef uint32_t result_type;
		static result_type min() { return 0; };
		static result_type max() { return std::numeric_limits<uint32_t>::max(); };
		result_type operator()();

	private:

		uint64_t state;
		uint64_t inc;
	};

	class Random
	{
	public:

		Random();
		explicit Random(uint64_t seed);

		void seed(uint64_t seed);

		int32_t getInt32(int32_t min = std::numeric_limits<int32_t>::lowest(), int32_t max = std::numeric_limits<int32_t>::max());
		uint32_t getUint32(uint32_t min = std::numeric_limits<uint32_t>::lowest(), uint32_t max = std::numeric_limits<uint32_t>::max());
		float getFloat();
		double getDouble();

		Color getColor(bool randomAlpha = false);
		Vector2 getVector2();
		Vector3 getVector3();

	private:

		RandomGeneratorPCG generator;
	};
}

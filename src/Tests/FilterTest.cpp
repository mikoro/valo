// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Filters/Filter.h"

using namespace Raycer;

TEST_CASE("Filter functionality", "[filter]")
{
	Filter filter;

	for (uint32_t k = 0; k <= 5; ++k)
	{
		filter.type = static_cast<FilterType>(k);

		std::ofstream file1(tfm::format("filter_%s_1D.txt", filter.getName()));
		std::ofstream file2(tfm::format("filter_%s_2D.txt", filter.getName()));

		float extent = 12.0f;
		uint32_t steps = 1000;
		float stepSize = extent / steps;

		for (uint32_t i = 0; i < steps; ++i)
		{
			float x = -(extent / 2.0f) + i * stepSize;
			file1 << tfm::format("%f %f\n", x, filter.getWeight(x));
		}

		steps = 40;
		stepSize = extent / steps;

		for (uint32_t i = 0; i < steps; ++i)
		{
			for (uint32_t j = 0; j < steps; ++j)
			{
				float x = -(extent / 2.0f) + j * stepSize;
				float y = -(extent / 2.0f) + i * stepSize;
				file2 << tfm::format("%f %f %f\n", x, y, filter.getWeight(Vector2(x, y)));
			}
		}

		file1.close();
		file2.close();
	}
}

#endif

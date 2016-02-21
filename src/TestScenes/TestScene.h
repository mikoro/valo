// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Scene;

	class TestScene
	{
	public:

		static Scene create(uint64_t number);
		static Scene create1();
		static Scene create2();
		static Scene create3();

		static const uint64_t TEST_SCENE_COUNT = 3;
	};
}

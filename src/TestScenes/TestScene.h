// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Valo
{
	class Scene;

	class TestScene
	{
	public:

		static Scene create(uint32_t number);
		static Scene create1();
		static Scene create2();
		static Scene create3();
		static Scene create4();
		static Scene create5();
		static Scene create6();
		static Scene create7();
		static Scene create8();

		static const uint32_t TEST_SCENE_COUNT = 8;
	};
}

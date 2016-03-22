// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Allocator
	{
	public:

		void* operator new(size_t len);
		void operator delete(void *ptr);

		static void* malloc(size_t len);
		static void free(void* ptr);
	};
}

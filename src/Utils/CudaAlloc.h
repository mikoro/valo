// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	template <typename T>
	class CudaAlloc
	{
	public:

		explicit CudaAlloc(bool pinned = false);
		~CudaAlloc();

		void resize(size_t count);
		void write(T* source, size_t count);

		CUDA_CALLABLE T* getPtr() const;
		T* getHostPtr() const;
		T* getDevicePtr() const;

	private:

		void release();

		bool pinned = false;

		T* hostPtr = nullptr;
		T* devicePtr = nullptr;
	};
}

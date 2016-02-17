// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <complex>

namespace Raycer
{
	template <uint64_t N>
	class Polynomial
	{
	public:

		Polynomial();
		explicit Polynomial(const float* coefficients);
		Polynomial& operator=(const Polynomial &other) = delete;

		void setCoefficients(const float* coefficients);

		std::complex<float> evaluate(const std::complex<float>& x) const;

		const std::complex<float>* findAllRoots(uint64_t maxIterations = 64, float changeThreshold = 0.0001f);
		const float* findAllPositiveRealRoots(uint64_t& count, uint64_t maxIterations = 64, float changeThreshold = 0.0001f, float imagZeroThreshold = 0.0001f);
		bool findSmallestPositiveRealRoot(float& result, uint64_t maxIterations = 64, float changeThreshold = 0.0001f, float imagZeroThreshold = 0.0001f);

	private:

		const uint64_t size = N;
		const uint64_t degree = N - 1;

		float coefficients[N];
		std::complex<float> roots[N - 1];
		std::complex<float> previousRoots[N - 1];
		float positiveRealRoots[N - 1];
	};
}

#include "Math/Polynomial.inl"

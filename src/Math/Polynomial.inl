// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Math/MathUtils.h"

namespace Raycer
{
	template <uint64_t N>
	Polynomial<N>::Polynomial()
	{
	}

	template <uint64_t N>
	Polynomial<N>::Polynomial(const float* coefficients_)
	{
		setCoefficients(coefficients_);
	}

	template <uint64_t N>
	void Polynomial<N>::setCoefficients(const float* coefficients_)
	{
		for (uint64_t i = 0; i < size; ++i)
			coefficients[i] = coefficients_[i];
	}

	template <uint64_t N>
	std::complex<float> Polynomial<N>::evaluate(const std::complex<float>& x) const
	{
		std::complex<float> y(0.0f, 0.0f);

		for (uint64_t i = 0; i < size; ++i)
			y += coefficients[i] * pow(x, int32_t(degree - i)); // raising to signed power is faster

		return y;
	}

	// Durand-Kerner algorithm
	// https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method
	template <uint64_t N>
	const std::complex<float>* Polynomial<N>::findAllRoots(uint64_t maxIterations, float changeThreshold)
	{
		// normalize (so that highest degree coeff is one)
		float denominator = coefficients[0];

		for (uint64_t i = 0; i < size; ++i)
			coefficients[i] /= denominator;

		// seed roots with distributed complex values
		std::complex<float> seed(0.4f, 0.9f);

		for (uint64_t i = 0; i < degree; ++i)
			roots[i] = previousRoots[i] = pow(seed, int32_t(i));

		for (uint64_t i = 0; i < maxIterations; ++i)
		{
			for (uint64_t j = 0; j < degree; ++j)
			{
				// numerator
				std::complex<float> temp = evaluate(roots[j]);

				// denominators
				for (uint64_t k = 0; k < degree; ++k)
				{
					if (k == j)
						continue;

					temp /= (roots[j] - roots[k]);
				}

				// rest of the equation
				roots[j] = roots[j] - temp;
			}

			// break iteration if roots are not changing much
			if (i >= 4)
			{
				bool notChanged = true;

				for (uint64_t j = 0; j < degree; ++j)
				{
					if (!MathUtils::almostSame(roots[j], previousRoots[j], changeThreshold))
					{
						notChanged = false;
						break;
					}
				}

				if (notChanged)
					break;
			}

			for (uint64_t j = 0; j < degree; ++j)
				previousRoots[j] = roots[j];
		}

		return roots;
	}

	template <uint64_t N>
	const float* Polynomial<N>::findAllPositiveRealRoots(uint64_t& count, uint64_t maxIterations, float changeThreshold, float imagZeroThreshold)
	{
		findAllRoots(maxIterations, changeThreshold);
		count = 0;

		for (uint64_t i = 0; i < degree; ++i)
		{
			if (MathUtils::almostZero(roots[i].imag(), imagZeroThreshold) && roots[i].real() > 0.0f)
			{
				positiveRealRoots[count] = roots[i].real();
				count++;
			}
		}

		return positiveRealRoots;
	}

	template <uint64_t N>
	bool Polynomial<N>::findSmallestPositiveRealRoot(float& result, uint64_t maxIterations, float changeThreshold, float imagZeroThreshold)
	{
		uint64_t count;
		findAllPositiveRealRoots(count, maxIterations, changeThreshold, imagZeroThreshold);

		if (count == 0)
			return false;

		std::sort(std::begin(positiveRealRoots), std::begin(positiveRealRoots) + count);
		result = positiveRealRoots[0];

		return true;
	}
}

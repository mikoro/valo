// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

namespace Raycer
{
	class Vector2;

	enum class FilterType { BOX, TENT, BELL, GAUSSIAN, MITCHELL, LANCZOS_SINC };

	class Filter
	{
	public:

		virtual ~Filter() {}

		virtual float getWeightX(float x) = 0;
		virtual float getWeightY(float y) = 0;

		float getWeight(float x, float y);
		float getWeight(const Vector2& point);
		
		float getRadiusX() const;
		float getRadiusY() const;
		Vector2 getRadius() const;

		static std::unique_ptr<Filter> getFilter(FilterType type);

	protected:

		float radiusX = 0.0f;
		float radiusY = 0.0f;
	};
}

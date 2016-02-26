// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Utils/PoissonDisc.h"

using namespace Raycer;

std::vector<Vector2> PoissonDisc::generate(uint64_t width, uint64_t height, float minDistance_, uint64_t iterationLimit, bool normalize)
{
	minDistance = minDistance_;
	cellSize = minDistance / float(M_SQRT2);
	gridWidth = uint64_t(ceil(float(width) / cellSize));
	gridHeight = uint64_t(ceil(float(height) / cellSize));
	
	grid.clear();
	grid.resize(gridHeight);

	for (auto& row : grid)
	{
		row.clear();
		row.resize(gridWidth);
	}

	points.clear();
	pointsToProcess.clear();

	Vector2 firstPoint = Vector2(random.getFloat() * float(width), random.getFloat() * float(height));

	points.push_back(firstPoint);
	pointsToProcess.push_back(firstPoint);

	PoissonDiscCell& cell1 = getCell(firstPoint);
	cell1.point = firstPoint;
	cell1.hasPoint = true;

	while (!pointsToProcess.empty())
	{
		Vector2 origin = getNextPointToProcess();

		for (uint64_t i = 0; i < iterationLimit; ++i)
		{
			Vector2 point = generateNewPoint(origin);

			if (point.x < 0.0f || point.x > float(width) || point.y < 0.0f || point.y > float(height))
				continue;

			if (isTooCloseToOthers(point))
				continue;

			points.push_back(point);
			pointsToProcess.push_back(point);

			PoissonDiscCell& cell2 = getCell(point);
			cell2.point = point;
			cell2.hasPoint = true;
		}
	}

	if (normalize)
	{
		for (Vector2& point : points)
			point /= Vector2(float(width), float(height));
	}
	
	return points;
}

PoissonDiscCellIndex PoissonDisc::getCellIndex(const Vector2& point)
{
	PoissonDiscCellIndex cellIndex;

	cellIndex.x = uint64_t(point.x / cellSize);
	cellIndex.y = uint64_t(point.y / cellSize);

	return cellIndex;
}

PoissonDiscCell& PoissonDisc::getCell(const Vector2& point)
{
	PoissonDiscCellIndex cellIndex = getCellIndex(point);
	return grid[cellIndex.y][cellIndex.x];
}

Vector2 PoissonDisc::getNextPointToProcess()
{
	uint64_t index = random.getUint64(0, pointsToProcess.size() - 1);

	Vector2 point = pointsToProcess[index];
	pointsToProcess.erase(pointsToProcess.begin() + index);

	return point;
}

Vector2 PoissonDisc::generateNewPoint(const Vector2& origin)
{
	float radius = minDistance * (1.0f + random.getFloat());
	float angle = 2.0f * float(M_PI) * random.getFloat();

	Vector2 point;

	point.x = origin.x + radius * std::cos(angle);
	point.y = origin.y + radius * std::sin(angle);

	return point;
}

bool PoissonDisc::isTooCloseToOthers(const Vector2& point)
{
	PoissonDiscCellIndex cellIndex = getCellIndex(point);

	uint64_t minx = std::max(int64_t(0), int64_t(cellIndex.x) - 2);
	uint64_t maxx = std::min(int64_t(gridWidth) - 1, int64_t(cellIndex.x) + 2);
	uint64_t miny = std::max(int64_t(0), int64_t(cellIndex.y) - 2);
	uint64_t maxy = std::min(int64_t(gridHeight) - 1, int64_t(cellIndex.y) + 2);

	float minDistance2 = minDistance * minDistance;

	for (uint64_t y = miny; y <= maxy; ++y)
	{
		for (uint64_t x = minx; x <= maxx; ++x)
		{
			PoissonDiscCell& cell = grid[y][x];

			if (!cell.hasPoint)
				continue;

			Vector2 pointToOther = cell.point - point;

			if (pointToOther.lengthSquared() < minDistance2)
				return true;
		}
	}

	return false;
}

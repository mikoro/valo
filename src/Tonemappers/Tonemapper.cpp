// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tonemappers/Tonemapper.h"
#include "Tonemappers/PassthroughTonemapper.h"
#include "Tonemappers/LinearTonemapper.h"
#include "Tonemappers/SimpleTonemapper.h"
#include "Tonemappers/ReinhardTonemapper.h"

using namespace Raycer;

std::unique_ptr<Tonemapper> Tonemapper::getTonemapper(TonemapperType type)
{
	switch (type)
	{
		case TonemapperType::PASSTHROUGH: return std::make_unique<PassthroughTonemapper>();
		case TonemapperType::LINEAR: return std::make_unique<LinearTonemapper>();
		case TonemapperType::SIMPLE: return std::make_unique<SimpleTonemapper>();
		case TonemapperType::REINHARD: return std::make_unique<ReinhardTonemapper>();
		default: throw std::runtime_error("Unknown tonemapper type");
	}
}

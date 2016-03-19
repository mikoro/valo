// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef RUN_UNIT_TESTS
#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#endif

#include "Core/App.h"

using namespace Raycer;

int main(int argc, char** argv)
{
#ifdef RUN_UNIT_TESTS
	return Catch::Session().run(argc, argv);
#else
	return App::run(argc, argv);
#endif
}

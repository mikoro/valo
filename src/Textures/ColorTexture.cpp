// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Textures/ColorTexture.h"
#include "Rendering/Color.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"

using namespace Raycer;

void ColorTexture::initialize(Scene& scene)
{
	(void)scene;
}

Color ColorTexture::getColor(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;
	(void)position;

	return color;
}

double ColorTexture::getValue(const Vector2& texcoord, const Vector3& position) const
{
	(void)texcoord;
	(void)position;

	return 0.0;
}

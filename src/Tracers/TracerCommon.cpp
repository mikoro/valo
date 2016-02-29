// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracers/TracerCommon.h"
#include "Math/Vector3.h"
#include "Rendering/Color.h"
#include "Tracing/Intersection.h"
#include "Materials/Material.h"
#include "Textures/Texture.h"

using namespace Raycer;

void TracerCommon::calculateNormalMapping(Intersection& intersection)
{
	Color normalColor = intersection.material->normalTexture->getColor(intersection.texcoord, intersection.position);
	Vector3 normal(normalColor.r * 2.0f - 1.0f, normalColor.g * 2.0f - 1.0f, normalColor.b * 2.0f - 1.0f);
	Vector3 mappedNormal = intersection.onb.u * normal.x + intersection.onb.v * normal.y + intersection.onb.w * normal.z;
	intersection.normal = mappedNormal.normalized();
}

Color TracerCommon::calculateNormalColor(const Vector3& normal)
{
	Color normalColor;

	normalColor.r = (normal.x + 1.0f) / 2.0f;
	normalColor.g = (normal.y + 1.0f) / 2.0f;
	normalColor.b = (normal.z + 1.0f) / 2.0f;

	return normalColor;
}

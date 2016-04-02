// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Intersection.h"
#include "Core/Scene.h"
#include "Materials/BlinnPhongMaterial.h"
#include "Materials/Material.h"
#include "Math/Mapper.h"
#include "Textures/Texture.h"
#include "Math/Random.h"

using namespace Raycer;

CUDA_CALLABLE Vector3 BlinnPhongMaterial::getDirection(const Material& material, const Intersection& intersection, Random& random)
{
	(void)material;

	return Mapper::mapToCosineHemisphere(random.getVector2(), intersection.onb);
}

CUDA_CALLABLE Color BlinnPhongMaterial::getBrdf(const Scene& scene, const Material& material, const Intersection& intersection, const Vector3& in, const Vector3& out)
{
	(void)in;
	(void)out;

	return material.getReflectance(scene, intersection.texcoord, intersection.position) / float(M_PI);
}

CUDA_CALLABLE float BlinnPhongMaterial::getPdf(const Material& material, const Intersection& intersection, const Vector3& out)
{
	(void)material;

	return intersection.normal.dot(out) / float(M_PI);
}

CUDA_CALLABLE Color BlinnPhongMaterial::getSpecularReflectance(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
	if (specularReflectanceTextureIndex != -1)
		return scene.getTexture(specularReflectanceTextureIndex).getColor(scene, texcoord, position);
	else
		return specularReflectance;
}

CUDA_CALLABLE Color BlinnPhongMaterial::getGlossiness(const Scene& scene, const Vector2& texcoord, const Vector3& position) const
{
	if (glossinessTextureIndex != -1)
		return scene.getTexture(glossinessTextureIndex).getColor(scene, texcoord, position);
	else
		return glossiness;
}

// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/ONB.h"
#include "Math/Matrix4x4.h"

using namespace Raycer;

const ONB ONB::UP = ONB(Vector3(-1.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, -1.0f), Vector3(0.0f, 1.0f, 0.0f));

ONB::ONB()
{
}

ONB::ONB(const Vector3& u_, const Vector3& v_, const Vector3& w_) : u(u_), v(v_), w(w_)
{
}

ONB ONB::transformed(const Matrix4x4& tranformation) const
{
	ONB result;

	result.u = tranformation.transformDirection(u).normalized();
	result.v = tranformation.transformDirection(v).normalized();
	result.w = tranformation.transformDirection(w).normalized();

	return result;
}

ONB ONB::fromNormal(const Vector3& normal, const Vector3& up)
{
	Vector3 u_ = normal.cross(up).normalized();
	Vector3 v_ = u_.cross(normal).normalized();
	Vector3 w_ = normal;

	return ONB(u_, v_, w_);
}

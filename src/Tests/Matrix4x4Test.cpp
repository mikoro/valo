// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#ifdef RUN_UNIT_TESTS

#include "catch/catch.hpp"

#include "Math/Matrix4x4.h"
#include "Math/Vector3.h"
#include "Math/EulerAngle.h"

using namespace Raycer;

TEST_CASE("Matrix4x4 functionality", "[matrix4x4]")
{
	Matrix4x4 m1 = Matrix4x4(
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f, 7.0f);

	Matrix4x4 m2 = Matrix4x4(
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f,
		9.0f, 1.0f, 2.0f, 3.0f,
		4.0f, 5.0f, 6.0f, 7.0f);

	Matrix4x4 m3 = Matrix4x4(
		9.0f, 8.0f, 7.0f, 6.0f,
		5.0f, 4.0f, 3.0f, 2.0f,
		1.0f, 2.0f, 3.0f, 4.0f,
		5.0f, 6.0f, 7.0f, 8.0f);

	Matrix4x4 m4 = Matrix4x4(
		9.0f, 8.0f, 7.0f, 0.0f,
		5.0f, 4.0f, 3.0f, 0.0f,
		1.0f, 2.0f, 3.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f);

	Vector3 v1 = Vector3(1.0f, 3.0f, 5.0f);

	REQUIRE(m1 == m2);
	REQUIRE(m1 != m3);

	REQUIRE((m1 + m2) == Matrix4x4(
		2.0f, 4.0f, 6.0f, 8.0f,
		10.0f, 12.0f, 14.0f, 16.0f,
		18.0f, 2.0f, 4.0f, 6.0f,
		8.0f, 10.0f, 12.0f, 14.0f));

	REQUIRE((m1 - m2) == Matrix4x4::ZERO);

	REQUIRE((m1 * 2.0f) == Matrix4x4(
		2.0f, 4.0f, 6.0f, 8.0f,
		10.0f, 12.0f, 14.0f, 16.0f,
		18.0f, 2.0f, 4.0f, 6.0f,
		8.0f, 10.0f, 12.0f, 14.0f));

	REQUIRE((2.0f * m1) == Matrix4x4(
		2.0f, 4.0f, 6.0f, 8.0f,
		10.0f, 12.0f, 14.0f, 16.0f,
		18.0f, 2.0f, 4.0f, 6.0f,
		8.0f, 10.0f, 12.0f, 14.0f));

	REQUIRE((m1 / 2.0f) == Matrix4x4(
		0.5f, 1.0f, 1.5f, 2.0f,
		2.5f, 3.0f, 3.5f, 4.0f,
		4.5f, 0.5f, 1.0f, 1.5f,
		2.0f, 2.5f, 3.0f, 3.5f));

	REQUIRE((m1 * m3) == Matrix4x4(
		42.0f, 46.0f, 50.0f, 54.0f,
		122.0f, 126.0f, 130.0f, 134.0f,
		103.0f, 98.0f, 93.0f, 88.0f,
		102.0f, 106.0f, 110.0f, 114.0f));

	REQUIRE((m1 * m2 * m3) == Matrix4x4(
		1003.0f, 1016.0f, 1029.0f, 1042.0f,
		2479.0f, 2520.0f, 2561.0f, 2602.0f,
		1012.0f, 1054.0f, 1096.0f, 1138.0f,
		2110.0f, 2144.0f, 2178.0f, 2212.0f));

	REQUIRE((m4.transformPosition(v1)) == Vector3(68.0f, 32.0f, 22.0f));

	REQUIRE((-m1) == Matrix4x4(
		-1.0f, -2.0f, -3.0f, -4.0f,
		-5.0f, -6.0f, -7.0f, -8.0f,
		-9.0f, -1.0f, -2.0f, -3.0f,
		-4.0f, -5.0f, -6.0f, -7.0f));

	REQUIRE(m1.transposed() == Matrix4x4(
		1.0f, 5.0f, 9.0f, 4.0f,
		2.0f, 6.0f, 1.0f, 5.0f,
		3.0f, 7.0f, 2.0f, 6.0f,
		4.0f, 8.0f, 3.0f, 7.0f));

	REQUIRE(Matrix4x4::scale(1.0f, 2.0f, 3.0f) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 2.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 3.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE(Matrix4x4::translate(1.0f, 2.0f, 3.0f) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f, 2.0f,
		0.0f, 0.0f, 1.0f, 3.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE((Matrix4x4::translate(1.0f, 2.0f, 3.0f) * Matrix4x4::scale(1.0f, 2.0f, 3.0f)) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 2.0f, 0.0f, 2.0f,
		0.0f, 0.0f, 3.0f, 3.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE((Matrix4x4::scale(1.0f, 2.0f, 3.0f) * Matrix4x4::translate(1.0f, 2.0f, 3.0f)) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 1.0f,
		0.0f, 2.0f, 0.0f, 4.0f,
		0.0f, 0.0f, 3.0f, 9.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE(Matrix4x4::rotateX(0) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE(Matrix4x4::rotateX(180.0f) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, -1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	REQUIRE(Matrix4x4::rotateX(90.0f) == Matrix4x4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, -1.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f));

	Vector3 scale = Vector3(2.0f, 3.0f, 4.0f);
	EulerAngle rotate = EulerAngle(10.0f, 20.0f, 30.0f);
	Vector3 translate = Vector3(50.0f, 60.0f, 70.0f);

	Matrix4x4 scaling = Matrix4x4::scale(scale);
	Matrix4x4 rotation = Matrix4x4::rotateXYZ(rotate.pitch, rotate.yaw, rotate.roll);
	Matrix4x4 translation = Matrix4x4::translate(translate);

	Matrix4x4 scalingInv = Matrix4x4::scale(scale.inversed());
	Matrix4x4 rotationInv = Matrix4x4::rotateZYX(-rotate.pitch, -rotate.yaw, -rotate.roll);
	Matrix4x4 translationInv = Matrix4x4::translate(-translate);

	Matrix4x4 transformation = translation * rotation * scaling;
	Matrix4x4 transformationInv = scalingInv * rotationInv * translationInv;
	Matrix4x4 transformationInv2 = transformation.inverted();

	REQUIRE(transformationInv == transformationInv2);

	Vector3 from(0.0f, 22.0f, 0.0f);
	Vector3 to(0.0f, 0.0f, 1.0f);
	rotation = Matrix4x4::rotate(from.normalized(), to.normalized());
	from = rotation.transformPosition(from);

	REQUIRE(from == Vector3(0.0f, 0.0f, 22.0f));
}

#endif

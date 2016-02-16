// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/Camera.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Settings.h"
#include "Math/MathUtils.h"
#include "Math/Vector2.h"
#include "Runners/WindowRunner.h"

using namespace Raycer;

void Camera::initialize()
{
	originalPosition = position;
	originalOrientation = orientation;
}

void Camera::setImagePlaneSize(uint64_t width, uint64_t height)
{
	imagePlaneWidth = double(width - 1);
	imagePlaneHeight = double(height - 1);
	aspectRatio = double(height) / double(width);
}

void Camera::reset()
{
	position = originalPosition;
	orientation = originalOrientation;
	velocity = Vector3(0.0, 0.0, 0.0);
	smoothVelocity = Vector3(0.0, 0.0, 0.0);
	smoothAcceleration = Vector3(0.0, 0.0, 0.0);
	angularVelocity = Vector3(0.0, 0.0, 0.0);
	smoothAngularVelocity = Vector3(0.0, 0.0, 0.0);
	smoothAngularAcceleration = Vector3(0.0, 0.0, 0.0);
}

void Camera::update(double timeStep)
{
	WindowRunner& windowRunner = App::getWindowRunner();
	Settings& settings = App::getSettings();
	MouseInfo mouseInfo = windowRunner.getMouseInfo();

	// LENS STUFF //

	if (windowRunner.keyWasPressed(GLFW_KEY_HOME))
	{
		if (projectionType == CameraProjectionType::PERSPECTIVE)
			projectionType = CameraProjectionType::ORTHOGRAPHIC;
		else if (projectionType == CameraProjectionType::ORTHOGRAPHIC)
			projectionType = CameraProjectionType::FISHEYE;
		else if (projectionType == CameraProjectionType::FISHEYE)
			projectionType = CameraProjectionType::PERSPECTIVE;
	}

	if (!windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) && !windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL) &&
		!windowRunner.keyIsDown(GLFW_KEY_LEFT_SHIFT) && !windowRunner.keyIsDown(GLFW_KEY_RIGHT_SHIFT))
	{
		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_DOWN))
		{
			if (projectionType == CameraProjectionType::PERSPECTIVE)
				fov -= 50.0 * timeStep;
			else if (projectionType == CameraProjectionType::ORTHOGRAPHIC)
				orthoSize -= 10.0 * timeStep;
			else if (projectionType == CameraProjectionType::FISHEYE)
				fishEyeAngle -= 50.0 * timeStep;
		}

		if (windowRunner.keyIsDown(GLFW_KEY_PAGE_UP))
		{
			if (projectionType == CameraProjectionType::PERSPECTIVE)
				fov += 50.0 * timeStep;
			else if (projectionType == CameraProjectionType::ORTHOGRAPHIC)
				orthoSize += 10.0 * timeStep;
			else if (projectionType == CameraProjectionType::FISHEYE)
				fishEyeAngle += 50.0 * timeStep;
		}
	}

	fov = std::max(1.0, std::min(fov, 180.0));
	orthoSize = std::max(0.0, orthoSize);
	fishEyeAngle = std::max(1.0, std::min(fishEyeAngle, 360.0));
	imagePlaneDistance = 0.5 / tan(MathUtils::degToRad(fov / 2.0));

	// SPEED MODIFIERS //

	if (windowRunner.keyWasPressed(GLFW_KEY_INSERT))
		cameraMoveSpeedModifier *= 2.0;

	if (windowRunner.keyWasPressed(GLFW_KEY_DELETE))
		cameraMoveSpeedModifier *= 0.5;

	double moveSpeed = settings.camera.moveSpeed * cameraMoveSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_CONTROL) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_CONTROL))
		moveSpeed *= settings.camera.slowSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_SHIFT) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_SHIFT))
		moveSpeed *= settings.camera.fastSpeedModifier;

	if (windowRunner.keyIsDown(GLFW_KEY_LEFT_ALT) || windowRunner.keyIsDown(GLFW_KEY_RIGHT_ALT))
		moveSpeed *= settings.camera.veryFastSpeedModifier;

	// ACCELERATIONS AND VELOCITIES //

	velocity = Vector3(0.0, 0.0, 0.0);
	angularVelocity = Vector3(0.0, 0.0, 0.0);
	bool movementKeyIsPressed = false;

	if (windowRunner.mouseIsDown(GLFW_MOUSE_BUTTON_LEFT) || settings.camera.freeLook)
	{
		smoothAngularAcceleration.y -= mouseInfo.deltaX * settings.camera.mouseSpeed;
		smoothAngularAcceleration.x += mouseInfo.deltaY * settings.camera.mouseSpeed;
		angularVelocity.y = -mouseInfo.deltaX * settings.camera.mouseSpeed;
		angularVelocity.x = mouseInfo.deltaY * settings.camera.mouseSpeed;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_W))
	{
		smoothAcceleration += forward * moveSpeed;
		velocity = forward * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_S))
	{
		smoothAcceleration -= forward * moveSpeed;
		velocity = -forward * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_D))
	{
		smoothAcceleration += right * moveSpeed;
		velocity = right * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_A))
	{
		smoothAcceleration -= right * moveSpeed;
		velocity = -right * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_E))
	{
		smoothAcceleration += up * moveSpeed;
		velocity = up * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_Q))
	{
		smoothAcceleration -= up * moveSpeed;
		velocity = -up * moveSpeed;
		movementKeyIsPressed = true;
	}

	if (windowRunner.keyIsDown(GLFW_KEY_SPACE) || !settings.camera.enableMovement)
	{
		velocity = Vector3(0.0, 0.0, 0.0);
		smoothVelocity = Vector3(0.0, 0.0, 0.0);
		smoothAcceleration = Vector3(0.0, 0.0, 0.0);
		angularVelocity = Vector3(0.0, 0.0, 0.0);
		smoothAngularVelocity = Vector3(0.0, 0.0, 0.0);
		smoothAngularAcceleration = Vector3(0.0, 0.0, 0.0);
	}

	// EULER INTEGRATION //

	cameraIsMoving = false;

	if (settings.camera.smoothMovement)
	{
		smoothVelocity += smoothAcceleration * timeStep;
		position += smoothVelocity * timeStep;

		smoothAngularVelocity += smoothAngularAcceleration * timeStep;
		orientation.yaw += smoothAngularVelocity.y * timeStep;
		orientation.pitch += smoothAngularVelocity.x * timeStep;

		cameraIsMoving = !smoothVelocity.isZero() || !smoothAngularVelocity.isZero();
	}
	else
	{
		position += velocity * timeStep;
		orientation.yaw += angularVelocity.y * timeStep;
		orientation.pitch += angularVelocity.x * timeStep;

		cameraIsMoving = !velocity.isZero() || !angularVelocity.isZero();
	}

	// DRAG & AUTO STOP //

	double smoothVelocityLength = smoothVelocity.length();
	double smoothAngularVelocityLength = smoothAngularVelocity.length();

	if ((smoothVelocityLength < settings.camera.autoStopSpeed * moveSpeed) && !movementKeyIsPressed)
		smoothVelocity = smoothAcceleration = Vector3(0.0, 0.0, 0.0);
	else if (!smoothVelocity.isZero())
		smoothAcceleration = settings.camera.moveDrag * (-smoothVelocity.normalized() * smoothVelocityLength);
	
	if (smoothAngularVelocityLength < settings.camera.autoStopSpeed * moveSpeed)
		smoothAngularVelocity = smoothAngularAcceleration = Vector3(0.0, 0.0, 0.0);
	else if (!smoothAngularVelocity.isZero())
		smoothAngularAcceleration = settings.camera.mouseDrag * (-smoothAngularVelocity.normalized() * smoothAngularVelocityLength);

	// ORIENTATION & BASIS VECTORS //

	orientation.clampPitch();
	orientation.normalize();

	ONB onb = ONB::fromNormal(orientation.getDirection());
	right = onb.u;
	up = onb.v;
	forward = onb.w;
	imagePlaneCenter = position + (forward * imagePlaneDistance);
}

bool Camera::isMoving() const
{
	return cameraIsMoving;
}

void Camera::saveState(const std::string& fileName) const
{
	App::getLog().logInfo("Saving camera state to %s", fileName);

	std::ofstream file(fileName);

	if (!file.is_open())
		throw std::runtime_error("Could not open file for writing camera state");

	file << tfm::format("x = %f", position.x) << std::endl;
	file << tfm::format("y = %f", position.y) << std::endl;
	file << tfm::format("z = %f", position.z) << std::endl;
	file << std::endl;
	file << tfm::format("pitch = %f", orientation.pitch) << std::endl;
	file << tfm::format("yaw = %f", orientation.yaw) << std::endl;
	file << tfm::format("roll = %f", orientation.roll) << std::endl;
	file << std::endl;
	file << tfm::format("scene.camera.position = Vector3(%.4f, %.4f, %.4f);", position.x, position.y, position.z) << std::endl;
	file << tfm::format("scene.camera.orientation = EulerAngle(%.4f, %.4f, %.4f);", orientation.pitch, orientation.yaw, orientation.roll) << std::endl;

	file.close();
}

Ray Camera::getRay(const Vector2& pixel, bool& isOffLens) const
{
	Ray ray;
	isOffLens = false;

	switch (projectionType)
	{
		case CameraProjectionType::PERSPECTIVE:
		{
			double dx = (pixel.x / imagePlaneWidth) - 0.5;
			double dy = (pixel.y / imagePlaneHeight) - 0.5;

			Vector3 imagePlanePixelPosition = imagePlaneCenter + (dx * right) + (dy * aspectRatio * up);

			ray.origin = position;
			ray.direction = (imagePlanePixelPosition - position).normalized();

		} break;

		case CameraProjectionType::ORTHOGRAPHIC:
		{
			double dx = (pixel.x / imagePlaneWidth) - 0.5;
			double dy = (pixel.y / imagePlaneHeight) - 0.5;

			ray.origin = position + (dx * orthoSize * right) + (dy * orthoSize * aspectRatio * up);
			ray.direction = forward;

		} break;

		// http://paulbourke.net/dome/fisheye/
		case CameraProjectionType::FISHEYE:
		{
			double dx = (pixel.x / imagePlaneWidth) * 2.0 - 1.0;
			double dy = (pixel.y / imagePlaneHeight) * 2.0 - 1.0;

			dx /= std::min(1.0, aspectRatio);
			dy *= std::max(1.0, aspectRatio);

			double r = sqrt(dx * dx + dy * dy);

			if (r > 1.0)
				isOffLens = true;

			double phi = atan2(dy, dx);
			double theta = r * (MathUtils::degToRad(fishEyeAngle) / 2.0);

			double u = sin(theta) * cos(phi);
			double v = sin(theta) * sin(phi);
			double w = cos(theta);

			ray.origin = position;
			ray.direction = u * right + v * up + w * forward;

		} break;

		default: break;
	}

	ray.precalculate();
	return ray;
}

Vector3 Camera::getRight() const
{
	return right;
}

Vector3 Camera::getUp() const
{
	return up;
}

Vector3 Camera::getForward() const
{
	return forward;
}

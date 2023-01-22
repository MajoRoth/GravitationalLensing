import math

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy

alpha = 1


def ode2D(x0, z0, theta, s=2000):
    theta = np.deg2rad(theta)

    y0 = [
        x0,
        z0,
        np.cos(theta),
        np.sin(theta)
    ]

    def f(y, t):
        def grad_ln_n(x, z):
            r_squared = x ** 2 + z ** 2
            return - alpha / (r_squared * (np.sqrt(r_squared) + alpha))

        grad_x_z = grad_ln_n(y[0], y[1])
        position_vec = np.array([y[0], y[1]])
        velocity_vec = np.array([y[2], y[3]])
        vx, vy = grad_x_z * position_vec - np.dot(grad_x_z * position_vec, velocity_vec) * velocity_vec

        return [y[2], y[3], vx, vy]

    t = np.arange(0, s, 1)
    y = odeint(f, y0, t)

    return y


def ode3D(x0, y0, z0, phi, theta, s=2000):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)

    y0 = [
        x0,
        y0,
        z0,
        np.sin(theta - math.pi / 2) * np.cos(phi),
        np.sin(theta - math.pi / 2) * np.sin(phi),
        np.cos(theta - math.pi / 2)
    ]

    def f(y, t):
        def grad_ln_n_2_masses(x, z, y, x1, y1, z1, x2, y2, z2):
            r1 = np.sqrt((x - x1) ** 2 + (z - z1) ** 2 + (y - y1) ** 2)
            r2 = np.sqrt((x - x2) ** 2 + (z - z2) ** 2 + (y - y2) ** 2)

            x_numerator = - alpha * ((x - x1) / r1 ** 3 + (x - x2) / r2 ** 3)
            y_numerator = - alpha * ((y - y1) / r1 ** 3 + (y - y2) / r2 ** 3)
            z_numerator = - alpha * ((z - z1) / r1 ** 3 + (z - z2) / r2 ** 3)
            denominator = (1 + alpha * (1 / r1 + 1 / r2))

            return np.array(
                [x_numerator / denominator, y_numerator / denominator, z_numerator / denominator]
            )

        grad_x_z = grad_ln_n_2_masses(x=y[0], y=y[1], z=y[2], x1=-20, y1=0, z1=0, x2=20, y2=0, z2=0)
        velocity_vec = np.array([y[3], y[4], y[5]])
        pos = grad_x_z - np.dot(grad_x_z, velocity_vec) * velocity_vec
        return [y[3], y[4], y[5],
                pos[0], pos[1], pos[2]]

    t = np.arange(0, s, 1)
    y = odeint(f, y0, t)

    return y


def approximated_analytical_angle(angle):
    b = 1000 * np.tan(np.deg2rad(angle))
    theta = 2 * alpha / b
    print("analytical theta {} and {} deg".format(theta, np.rad2deg(theta)))
    return theta


def error(real, approx):
    return 100 * abs(real - approx) / abs(real)


def calculate_angles(x, z):
    angles = np.arctan2(
        np.array([z[1] - z[0], z[-1] - z[-2]]),
        np.array([x[1] - x[0], x[-1] - x[-2]])
    )
    mid_angle = abs(angles[1]) - abs(angles[0])
    return angles[0], angles[1] - math.pi / 2, mid_angle


def einstein_ring_2D(x0=0, z0=1000, x1=0, z1=-500, theta_range=[]):
    def calculate_distance_route_from_point_2D(x, z, x1, z1):
        for p in zip(x, z):
            if (p[0] - x1) ** 2 + (p[1] - z1) ** 2 < 1:
                return True

        return False

    routes = list()
    for theta in theta_range:
        y = ode2D(x0=x0, z0=z0, theta=theta, s=2000)
        x, z, _, _ = zip(*y)
        if calculate_distance_route_from_point_2D(x, z, x1, z1):
            routes.append(
                [x, z]
            )
    return routes


def einstein_ring_3D(x0=0, y0=0, z0=1000, x1=0, y1=0, z1=-1300, theta_range=[], phi_range=[]):
    def calculate_distance_route_from_point_3D(x, y, z, x1, y1, z1):
        for p in zip(x, y, z):
            if (p[0] - x1) ** 2 + (p[1] - y1) ** 2 + (p[2] - z1) ** 2 < 1:
                return True
        return False

    routes = list()

    for theta in theta_range:
        for phi in phi_range:
            route = ode3D(x0=x0, y0=y0, z0=z0, theta=theta, phi=phi, s=4000)
            x, y, z, _, _, _ = zip(*route)
            print("Calculated a Route with theta = {} and phi = {}".format(theta, phi))

            if calculate_distance_route_from_point_3D(x, y, z, x1, y1, z1):
                routes.append(
                    [x, y, z]
                )

    return routes


def create_x_z_rays(routes, phi_range):
    new_routes = []
    for route in routes:
        for phi in phi_range:
            x = list(route[0]) * np.full(len(route[0]), np.cos(np.deg2rad(phi)))
            z = route[1]
            new_routes.append([x, z])

    return new_routes


def create_x_y_points_2D(routes, phi_range):
    circle_points_x = []
    circle_points_y = []
    for route in routes:
        in_angle, out_angle, mid_angle = calculate_angles(route[0], route[1])
        for phi in phi_range:
            x = np.tan(out_angle) * 500 * np.cos(phi)
            y = np.tan(out_angle) * 500 * np.sin(phi)
            circle_points_x.append(x)
            circle_points_y.append(y)

    return circle_points_x, circle_points_y


def create_x_y_points_3D(routes):
    def point_in_z_plane(route):
        """
        returns the x,y of the neerest point to z=0 plane
        :param route:
        :return:
        """
        min_dist = 100
        min_point = None
        for point in zip(*route):
            if point[2] < min_dist:
                min_dist = point[2]
                min_point = point
        return min_point[0], min_point[1]

    circle_points_x = []
    circle_points_y = []
    for route in routes:
        x, y = point_in_z_plane(route)
        circle_points_x.append(x)
        circle_points_y.append(y)

    return circle_points_x, circle_points_y


def section_1():
    angle = 275
    y = ode2D(x0=0, z0=1000, theta=angle, s=2500)
    x, z, _, _ = zip(*y)
    plt.rcParams['text.usetex'] = True
    plt.plot(x, z)

    theta_in, theta_out, theta_mid = calculate_angles(x, z)
    theta_approx = approximated_analytical_angle(angle - 270)
    print("The error of the approximation is {} %".format(error(theta_mid, theta_approx)))
    plt.scatter([0], [0])
    plt.title("Ray Projectile With a={} degrees".format(angle))
    plt.grid(True)
    plt.savefig('Ray trace')
    plt.show()


def section_2():
    phi_range = np.linspace(0, 360, 25)
    theta_range = np.linspace(271, 272, 11)

    routes = einstein_ring_2D(theta_range=theta_range)
    x_z_rays = create_x_z_rays(routes, phi_range)
    x_y_points = create_x_y_points_2D(routes, phi_range)

    for ray in x_z_rays:
        plt.plot(ray[0], ray[1])
    plt.grid(True)
    plt.title("Einsteing Ring x-z plane")
    plt.savefig('Einsteing Ring x-z plane')
    plt.show()

    plt.scatter(x=x_y_points[0], y=x_y_points[1])
    plt.grid(True)
    plt.title("Einsteing Ring x-y view")
    plt.savefig('x-y circle')
    plt.show()


def section_3():
    y = ode3D(x0=0, y0=0, z0=1000, theta=272, phi=10, s=1500)
    x, y, z, _, _, _ = zip(*y)
    plt.rcParams['text.usetex'] = True
    plt.plot(x, z)
    plt.scatter([-20, 20], [0, 0])
    plt.grid(True)
    plt.title('Ray trace with 2 masses')
    plt.show()

    theta_range = np.linspace(270, 280, 21)
    phi_range = np.linspace(0, 360, 9)

    routes = einstein_ring_3D(theta_range=theta_range, phi_range=phi_range)
    for ray in routes:
        plt.plot(ray[0], ray[2])

    plt.title('2 masses x-z view')
    plt.show()

    x_points, y_points = create_x_y_points_3D(routes)
    plt.scatter(x=x_points, y=y_points)
    plt.grid(True)
    plt.title("2 masses x-y view")
    plt.show()


if __name__ == '__main__':
    section_1()
    section_2()
    section_3()

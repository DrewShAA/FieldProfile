import math
import sys


def calculate_angle_btw_plane(a1=0., b1=0., c1=0.,
                              a2=0., b2=0., c2=0.):
    cos_alpha = abs(a1 * a2 + b1 * b2 + c1 * c2) / \
                (math.sqrt(a1 ** 2 + b1 ** 2 + c1 ** 2) *
                 math.sqrt(a2 ** 2 + b2 ** 2 + c2 ** 2))
    alpha_rad = math.acos(cos_alpha)
    return alpha_rad * 180 / math.pi


def main():
    print(calculate_angle_btw_plane(a1=0.02, b1=-0.02, c1=-1, c2=1))


if __name__ == '__main__':
    main()

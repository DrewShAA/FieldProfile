from dataclasses import dataclass
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())

@dataclass
class GeoPoint:
    x: float
    y: float
    z: int


def parce(line: str):
    z = int(line.split(',')[0])
    x, y = line.split('(')[-1].split()
    x = float(x)
    y = float(y.split(')')[0])
    return x, y, z


def get_datas(file_name):

    all_points = []
    with open(file_name, 'r') as file:
        _ = file.readline()

        for line in file:
            all_points.append(GeoPoint(*parce(line)))
    # print(*all_points, sep='\n')
    return all_points


def main():
    points = get_datas('samples/example_1/Results_elevation_ex_1.csv')
    print(points)


if __name__ == '__main__':
    main()

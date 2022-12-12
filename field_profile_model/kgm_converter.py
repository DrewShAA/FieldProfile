import json

import kml2geojson
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())


def main():
    dict_information = kml2geojson.main.convert('samples/Пример_1_ed.kml')[0]
    # with open('json_files/style.json', 'r') as file:
    #     dict_information = json.load(file)
    kgm_elements = dict_information.get('features')
    if_first = True

    left = None
    right = None
    bottom = None
    top = None

    for element in kgm_elements:
        if 'geometry' not in element.keys():
            continue
        if element.get('geometry').get('type') != "Polygon":
            continue
        if if_first:
            if_first = False
            left = right = element.get('geometry').get('coordinates')[0][0][0]
            bottom = top = element.get('geometry').get('coordinates')[0][0][1]

        for x, y, z in element.get('geometry').get('coordinates')[0]:
            left = x if x < left else left
            right = x if x > right else right
            top = y if y > top else top
            bottom = y if y < bottom else bottom
    print(left, top, right, bottom)
    return left, top, right, bottom


if __name__ == "__main__":
    main()

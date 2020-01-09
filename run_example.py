import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import polygon_operations as polop
from shapely.geometry import Polygon

# Polygon with redundant points
redundant_polygon = [(0,0), (1,0), (2,0), (2,1), (2,2), (1,2), (0,2), (0,1), (0,0)]
polop.plot_polygons([redundant_polygon])
erased_points, new_polygon = polop.simplify_polygon(redundant_polygon, criteria=0, direct_erase_limit=0, longlat=False)
polop.plot_polygons([new_polygon])

# Polygon with non signiticant points
n = 100
x = np.linspace(2,0, n)
points = list(zip(0.001*np.cos(np.pi*np.arange(n)), x))
too_complicated_polygon = [(0,0), (2,0), (2,1), (2,2), (1,2), (0,2)] + points
polop.plot_polygons([too_complicated_polygon])
erased_points, less_coplicated_polygon = polop.simplify_polygon(too_complicated_polygon, criteria=0.01, direct_erase_limit=0.01, longlat=False)
polop.plot_polygons([less_coplicated_polygon])

# valid polygon from non-valid polygon
non_valid_polygon = [(0,0), (1,0), (2,1.5), (3,0), (4,1), (0,0)]
polop.plot_polygons([non_valid_polygon])
worked, valid_polygon = polop.create_valid_polygon(non_valid_polygon, longlat=False, critical_distance_line_crosspoint=0.1, minor_distance=0.0001)
print('Inital polygon: validity: ' + str(Polygon(non_valid_polygon).is_valid) + '; calculated area: {:.2f}'.format(Polygon(non_valid_polygon).area))
print('Treated polygon: validity: ' + str(Polygon(valid_polygon).is_valid) + ';  calculated area: {:.2f}'.format(Polygon(valid_polygon).area))

# Real world case
badly_created_polygon = { "type": "Feature", "properties": { "idp": 0, 'name': 'bad polygon' }, "geometry": { "type": "Polygon", "coordinates": [ [ [ -0.858375044261295, 0.223157527006128 ], [ -0.858170588596769, 0.226991070715981 ], [ -0.856023804119251, 0.226991070715981 ], [ -0.856432715448302, 0.223310868754522 ], [ -0.853161424815895, 0.223361982670653 ], [ -0.853110310899763, 0.227042184632112 ], [ -0.854592614467573, 0.226991070715981 ], [ -0.854592614467573, 0.223157527006128 ], [ -0.858375044261295, 0.223106413089997 ], [ -0.858375044261295, 0.223208640922259 ], [ -0.858375044261295, 0.223259754838391 ], [ -0.858630613841951, 0.223106413089997 ], [ -0.858630613841951, 0.223106413089997 ], [ -0.858477272093557, 0.223055299173865 ], [ -0.858477272093557, 0.223055299173865 ], [ -0.858375044261295, 0.223055299173865 ], [ -0.858375044261295, 0.223055299173865 ], [ -0.858375044261295, 0.223055299173865 ], [ -0.858375044261295, 0.223106413089997 ], [ -0.858375044261295, 0.223106413089997 ], [ -0.858355876542745, 0.223130372738183 ], [ -0.858371849641536, 0.223095231920843 ], [ -0.858362265782262, 0.223095231920843 ], [ -0.858349487303229, 0.223104815780118 ], [ -0.858362265782262, 0.223120788878909 ], [ -0.858381433500811, 0.22311759425915 ], [ -0.858381433500811, 0.223111205019634 ], [ -0.858381433500811, 0.223111205019634 ], [ -0.858375044261295, 0.223157527006128 ] ] ] } }
polop.plot_parcels_dict([badly_created_polygon])
bad_polygons = [badly_created_polygon['geometry']['coordinates'][0], ]
# Increase the criteria value to match the problem scale
# Feel free to zoom in the lower left corner to see that happened
good_polygons = polop.simplyfy_and_correct_polygons(bad_polygons, criteria=100)
print('Inital polygon: validity: ' + str(Polygon(bad_polygons[0]).is_valid) + '; calculated area: {:.2f}ha'.format(polop.polygon_to_hectare(bad_polygons[0])))
print('Treated polygon: validity: ' + str(Polygon(good_polygons[0]).is_valid) + '; calculated area: {:.2f}ha'.format(polop.polygon_to_hectare(good_polygons[0])))

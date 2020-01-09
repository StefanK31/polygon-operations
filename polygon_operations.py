import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import pyproj
from shapely.geometry import Polygon, LineString
import utm


def simplify_polygon(initial_polygon, criteria=1, showplot=False, direct_erase_limit=0.01, longlat=True):
    """This function simplifies a polygon by erasing non-necessary points. This is cone by a loop deleting point
     by point based on a criteria that may be changed in the argument.
     Criteria values and direct_erase_limit are of the order of the area that may be neglected.
     These values need to be adapted to the desired precision.
     Accepts a polygon of the form [[p1_long, p1_lat], [p2_long, p2_lat], []...]. It returns the number of corrected
     points and the corrected polygon in the same format in a form of a numpy array.
     """
    def point_importance(polygon):
        """Calculates importance for all points"""
        angles = all_angles(polygon)
        dists = total_dists(polygon)
        criteria1 = np.array([dist[0] * dist[1] * np.sin(abs(angle / 180 * np.pi)) for dist, angle in zip(dists, angles)])
        criteria2 = all_point_line_dis(polygon)
        final_criteria = criteria1 * criteria2
        return np.array(final_criteria) ** (2/3)

    def angle(p_b, p, p_a):
        """gives angle difference for three seubsequent points"""
        diff_b = p - p_b
        diff_a = p_a - p
        if diff_b[1]==0:
            angle_b = np.pi/2
        else:
            angle_b = np.arctan(diff_b[0] / diff_b[1]) * 180 / np.pi
        if diff_a[1]==0:
            angle_a = np.pi/2
        else:
            angle_a = np.arctan(diff_a[0] / diff_a[1]) * 180 / np.pi
        return angle_a - angle_b

    def all_angles(polygon):
        """Calculates all inner angles of the polygon"""
        n = len(polygon)
        angles = []
        for i in range(len(polygon)):
            p_b = polygon[(i - 1) % n]
            p = polygon[(i) % n]
            p_a = polygon[(i + 1) % n]
            angles.append(angle(p_b, p, p_a))
        return angles

    def all_point_line_dis(polygon):
        """calculates the distance between every point to the imagined line between previous and subsequent points"""
        n = len(polygon)
        dis_pl = []
        for i in range(len(polygon)):
            p_b = polygon[(i - 1) % n]
            p = polygon[(i) % n]
            p_a = polygon[(i + 1) % n]
            ang_b = angle(p_a, p_b, p, )
            height = np.sin(np.abs(np.pi / 180 * ang_b)) * distance_space(p_b[0], p[0], p_b[1], p[1])
            dis_pl.append(height)
        return np.array(dis_pl)

    def total_dists(polygon):
        """calculates the distance for all points to its neighbors"""
        n = len(polygon)
        dist = []
        for i in range(len(polygon)):
            p_b = polygon[(i - 1) % n]
            p = polygon[(i) % n]
            p_a = polygon[(i + 1) % n]
            dist.append([distance_space(p_b[0], p[0], p_b[1], p[1]),
                         distance_space(p_a[0], p[0], p_a[1], p[1])])
        return dist

    zone = {}
    polygon, shiftpoint = transform_polygon(initial_polygon, angle=0, params=zone, longlat=longlat)
    initial_area = Polygon(polygon[:]).area
    if showplot:
        plot_polygons([polygon])
    if all(polygon[0] == polygon[-1]):
        polygon = np.delete(polygon, -1, 0)
    # direct point suppression
    erased_points = 0
    quick_correction_finished = False
    while not quick_correction_finished:
        factors = point_importance(polygon)
        to_erase = [i for i, fac in enumerate(factors) if fac <= direct_erase_limit]
        if (len(factors) - len(to_erase)) <= 3:
            to_erase = [i for i, fac in enumerate(factors[:-2]) if fac <= direct_erase_limit]
        if len(to_erase) <= 2:
            quick_correction_finished = True
        for j, i in enumerate(reversed(to_erase)):
            if (j % 2):
                # print('exp iteration: erase minimum of {:.2f}'.format(factors[i]))
                if showplot:
                    plt.plot(polygon[i][0], polygon[i][1], 'go')
                polygon = np.delete(polygon, i, 0)
                erased_points += 1
    # iterative point suppression
    correction_finished = False
    while not correction_finished:
        factors = point_importance(polygon)
        minimum = np.min(factors)
        if (minimum <= criteria):
            # print('lin iteration: erase minimum of {:.2f}'.format(minimum))
            if showplot:
                plt.plot(polygon[factors.argmin()][0], polygon[factors.argmin()][1], 'go')
                plt.text(polygon[factors.argmin()][0], polygon[factors.argmin()][1], '{:.2f}'.format(minimum))
            polygon = np.delete(polygon, factors.argmin(), 0)
            erased_points += 1
            if all(polygon[0] == polygon[-1]):
                polygon = np.delete(polygon, -1, 0)
        else:
            correction_finished = True
        if len(polygon[0]) == 3:
            correction_finished = True
    if showplot:
        plt.plot(np.array(polygon).T[0], np.array(polygon).T[1], 'r')
    final_area = Polygon(polygon[:]).area
    # print('final area: {:.2f}; initial area: {:.2f}; {:.4}% of the area is conserved'.format(final_area, initial_area, 100*final_area / initial_area))
    valid_polygon, _ = transform_polygon(polygon, angle=0, params=zone, shift_point=shiftpoint, inverse=True, longlat=longlat)
    return erased_points, valid_polygon


def create_valid_polygon(initial_polygon, critical_distance_line_crosspoint=1, critical_area_additional_polygon=0.1,
                         execute_twice=False, minor_distance=0.001, longlat=True):
    """Tries to create a valid polygon from a non valid one. Expects an initial polygon in longtitude-latiude
    returns polygon in longitude-latitude representation. minor_distance is problem and precision relavant.
    critical_area_additional_polygon is a surface area under that areas may be neglected"""
    def avoid_all_crosspoints(ini_polygon):
        """This function recursively divides the polygon at crosspoints is subpolygons and then merges them together
        at new points next to the crosspoint"""
        def merge_polygons(poly1, poly2):
            # plot_polygons([poly1, poly2])
            # plt.title('polygons to merge')
            direction1 = (poly2[0] - poly2[-2]) / distance_space(poly2[0][0], poly2[-2][0], poly2[0][1], poly2[-2][1])
            direction2 = (poly2[0] - poly2[+1]) / distance_space(poly2[0][0], poly2[+1][0], poly2[0][1], poly2[+1][1])
            sum_direction = direction1 + direction2
            sum_direction_norm = sum_direction / distance_space(0, sum_direction[0], 0, sum_direction[1])
            orthogonal_direction = np.array([-sum_direction_norm[1], sum_direction_norm[0]])
            crosspoint_both = poly2[0]
            wheretuple = np.where(poly1 == crosspoint_both)
            poly1_crosspoint_index = [i[0] for i in wheretuple if i[0] == i[1]][0]
            newP1 = crosspoint_both + minor_distance * 0.05 * orthogonal_direction
            newP2 = crosspoint_both - minor_distance * 0.05 * orthogonal_direction
            new_poly2 = poly2[:].copy()
            new_poly2[0] = newP1
            new_poly2[-1] = newP2
            reversed_order_poly2 = new_poly2[::-1]
            merged_polygon1 = np.append(poly1[0:poly1_crosspoint_index], reversed_order_poly2, axis=0)
            merged_polygon1 = np.append(merged_polygon1, poly1[poly1_crosspoint_index + 1:], axis=0)
            merged_polygon2 = np.append(poly1[0:poly1_crosspoint_index], new_poly2, axis=0)
            merged_polygon2 = np.append(merged_polygon2, poly1[poly1_crosspoint_index + 1:], axis=0)
            new_poly2[0] = newP2
            new_poly2[-1] = newP1
            reversed_order_poly2 = new_poly2[::-1]
            merged_polygon3 = np.append(poly1[0:poly1_crosspoint_index], reversed_order_poly2, axis=0)
            merged_polygon3 = np.append(merged_polygon3, poly1[poly1_crosspoint_index + 1:], axis=0)
            merged_polygon4 = np.append(poly1[0:poly1_crosspoint_index], new_poly2, axis=0)
            merged_polygon4 = np.append(merged_polygon4, poly1[poly1_crosspoint_index + 1:], axis=0)
            if Polygon(merged_polygon1).is_valid:
                merged_polygon = merged_polygon1
            elif Polygon(merged_polygon2).is_valid:
                merged_polygon = merged_polygon2
            elif Polygon(merged_polygon3).is_valid:
                merged_polygon = merged_polygon3
            else:
                merged_polygon = merged_polygon4
            # plot_polygons([merged_polygon])
            # plt.title('Merged Polygon')
            return merged_polygon

        def search_crosspoint_and_devide(big_polygon):
            lines = []
            for i in range(len(big_polygon) - 1):
                lines.append(LineString([big_polygon[i], big_polygon[i + 1]]))
            for i in range(len(lines)):
                lines_to_check = list(range(len(lines)))
                lines_to_check.remove((i - 1) % len(lines))
                lines_to_check.remove((i) % len(lines))
                lines_to_check.remove((i + 1) % len(lines))
                for j in lines_to_check:
                    if lines[i].intersects(lines[j]):
                        crosspoint = lines[i].intersection(lines[j])
                        corsspoint_coords = [crosspoint.coords.xy[0][0], crosspoint.coords.xy[1][0]]
                        m = np.min([i, j])
                        n = np.max([i, j])
                        p1 = np.append(big_polygon[0:m + 1], [corsspoint_coords], axis=0)
                        p1 = np.append(p1, big_polygon[n + 1:], axis=0)
                        p2 = np.append([corsspoint_coords], big_polygon[m + 1:n + 1], axis=0)
                        p2 = np.append(p2, [corsspoint_coords], axis=0)
                        # plot_polygons([p1,p2])
                        # plt.title('divided polygons')
                        devided = True
                        return devided, [p1, p2]
            devided = False
            return devided, big_polygon

        devided, polygonarray = search_crosspoint_and_devide(ini_polygon)
        if devided:
            p1 = avoid_all_crosspoints(polygonarray[0])
            p2 = avoid_all_crosspoints(polygonarray[1])
            return merge_polygons(p1, p2)
        else:
            return ini_polygon
        return polygon

    if Polygon(initial_polygon).is_valid:
        return True, initial_polygon
    zone = {}
    polygon, shiftpoint = transform_polygon(initial_polygon, angle=0, params=zone, longlat=longlat)
    _, polygon = simplify_polygon(polygon, criteria=0, showplot=False, direct_erase_limit=0, longlat=False)
    plot_polygons([polygon])
    plt.title('validity: ' + str(Polygon(polygon).is_valid))
    initial_area = Polygon(polygon[:]).area
    # Check for isolated points
    if np.all(polygon[0] == polygon[-1]):
        polygon = polygon[:-1]
    i = 0
    while i < len(polygon):
        if np.all(polygon[i] == polygon[(i + 2) % len(polygon)]):
            print(i)
            to_erase = [i, (i + 1) % len(polygon)]  # erase in right order from backwards
            polygon = np.delete(polygon, np.max(to_erase), 0)
            polygon = np.delete(polygon, np.min(to_erase), 0)
            i = 0
        elif np.all(polygon[i] == polygon[(i + 1) % len(polygon)]):
            polygon = np.delete(polygon, (i + 1) % len(polygon), 0)
            i = 0
        else:
            i += 1
    if Polygon(polygon).is_valid:
        final_area = Polygon(polygon).area
        plot_polygons([polygon])
        plt.title('Valid polygon')
        print('Final area: {:.5f} vs initial area: {:.5f}'.format(final_area, initial_area))
        valid_polygon, _ = transform_polygon(polygon, angle=0, params=zone, shift_point=shiftpoint, inverse=True, longlat=longlat)
        return True, valid_polygon
    # Make sure last point = first point
    if not np.all(polygon[0] == polygon[-1]):
        polygon = np.append(polygon, [polygon[0]], axis=0)
    # Determine isolated areas for all points. If they are non-significant with respect to
    # critical_distance_line_crosspoint or critical_area_additional_polygon they will be erased
    point_runner = 0
    while point_runner < len(polygon):
        # Create lines
        lines = []
        for i in range(len(polygon) - 1):
            lines.append(LineString([polygon[i], polygon[i + 1]]))
        # Search cross points between one and the second subsequent line:
        for i in range(0, len(lines)):
            if lines[i].intersects(lines[(i + 2) % len(lines)]):
                crosspoint = lines[i].intersection(lines[(i + 2) % len(lines)])
                corsspoint_coords = [crosspoint.coords.xy[0][0], crosspoint.coords.xy[1][0]]
                distance_point_to_lines = crosspoint.distance(lines[(i + 1) % len(lines)])
                additional_polygon_area = Polygon(
                    [corsspoint_coords, polygon[(i + 1) % len(lines)], polygon[(i + 2) % len(lines)]]).area
                if distance_point_to_lines < critical_distance_line_crosspoint or additional_polygon_area < critical_area_additional_polygon:
                    new_polygon = polygon[0:i + 1].tolist() + [corsspoint_coords] + polygon[i + 3:].tolist()
                    # plot_polygons([new_polygon])
                    polygon = np.array(new_polygon)
                    point_runner = 0
                    break
        point_runner += 1
    if Polygon(polygon).is_valid:
        final_area = Polygon(polygon).area
        plot_polygons([polygon])
        plt.title('Valid polygon')
        print('Final area: {:.5f} vs initial area: {:.5f}'.format(final_area, initial_area))
        valid_polygon, _ = transform_polygon(polygon, angle=0, params=zone, shift_point=shiftpoint, inverse=True, longlat=longlat)
        return True, valid_polygon
    # Check all cross points and avoid them
    # minor shift double points
    for i in range(len(polygon)):
        points_to_check = list(range(len(polygon)))
        points_to_check.remove(i)
        for j in points_to_check:
            if np.all(polygon[i] == polygon[j]):
                if i == 0 and j == len(polygon) - 1:
                    pass
                elif j == 0 and i == len(polygon) - 1:
                    pass
                else:
                    polygon[j] = polygon[j] + minor_distance
    # Strategy: recursevly subdivide polygons and merge again
    new_polygon = avoid_all_crosspoints(polygon)
    new_polygon = avoid_all_crosspoints(np.flip(new_polygon, axis=0))
    if Polygon(new_polygon).is_valid:
        final_area = Polygon(new_polygon).area
        plot_polygons([new_polygon])
        plt.title('Valid polygon')
        print('Final area: {:.5f} vs initial area: {:.5f}'.format(final_area, initial_area))
        valid_polygon, _ = transform_polygon(new_polygon, angle=0, params=zone, shift_point=shiftpoint, inverse=True, longlat=longlat)
        return True, valid_polygon
    non_valid_polygon, _ = transform_polygon(polygon, angle=0, params=zone, shift_point=shiftpoint, inverse=True, longlat=longlat)
    if execute_twice:
        _, final_polygon = create_valid_polygon(non_valid_polygon, execute_twice=False, longlat=longlat)
    else:
        final_polygon = non_valid_polygon
    if Polygon(final_polygon).is_valid:
        plot_polygons([final_polygon])
        plt.title('Valid polygon')
        return True, final_polygon
    else:
        print("Polygon could not be corrected")
        return False, final_polygon


def simplyfy_and_correct_polygons(polygonlist,
                                  criteria=1,
                                  critical_distance_line_crosspoint=1,
                                  critical_area_additional_polygon=1,
                                  execute_twice=False):
    """Takes a list of polygons [[(long1, lat1), (long2, lat2), ...], ...], simplifies them and possible creates a valid polygon.
    Criteria values and direct_erase_limit are of the order of the area that may be neglected.
    These values need to be adapted to the desired precision.
    For more information, see simplify_polygon and create_valid_polygon
    """
    new_polygons = []
    for poly in polygonlist:
        long_lat = simplify_polygon(poly, criteria=criteria, showplot=False)[1]
        if not Polygon(long_lat.tolist()).is_valid:
            success, long_lat = create_valid_polygon(long_lat.tolist(),
                                                     critical_distance_line_crosspoint=critical_distance_line_crosspoint,
                                                     critical_area_additional_polygon=critical_area_additional_polygon,
                                                     execute_twice=execute_twice)
        new_polygons.append(long_lat.tolist())
    return new_polygons


def zone(point_in_long_lat):
    """ returns time zone as int"""
    easting, northing, zone_int, zoneletter = utm.from_latlon(point_in_long_lat[1], point_in_long_lat[0])
    return zone_int


def rotate_origin(coor, rot_angle):
    """Rotates coords around an rotation_angle counterclockwise"""
    rot_angle = np.deg2rad(rot_angle)
    xnew = + np.cos(rot_angle) * coor[0] + np.sin(rot_angle) * coor[1]
    ynew = - np.sin(rot_angle) * coor[0] + np.cos(rot_angle) * coor[1]
    return [xnew, ynew]


def transform_coordinates(x, y, angle=0, params={}, shift_point=False, inverse=False):
    """Transforms coordinates from longitude, latitude to meter and back again. For back transformation (inverse=True)
    a shift_point is necessary that is returned in the first transformation"""
    if not inverse:
        long = x
        lat = y
        if not 'zone' in params.keys():
            params['zone'] = zone([long[0], lat[0]])
        p = pyproj.Proj(proj='utm', zone=params['zone'], ellps='WGS84')
        x_in_meter, y_in_meter = p(long, lat)
        if type(shift_point)==bool:
            shift_point = np.mean([np.array(x_in_meter), np.array(y_in_meter)], axis=1)
        x_shifted = np.array(x_in_meter) - shift_point[0]
        y_shifted = np.array(y_in_meter) - shift_point[1]
        coords_rot = rotate_origin([x_shifted, y_shifted], angle)
        return coords_rot[0], coords_rot[1], shift_point
    elif inverse:
        x_rot = np.array(x)
        y_rot = np.array(y)
        coords_shifted = rotate_origin([x_rot, y_rot], -angle)
        x_in_meter = coords_shifted[0] + shift_point[0]
        y_in_meter = coords_shifted[1] + shift_point[1]
        p = pyproj.Proj(proj='utm', zone=params['zone'], ellps='WGS84')
        long, lat = p(x_in_meter, y_in_meter, inverse=True)
        return np.array(long), np.array(lat)


def transform_polygon(initial_polygon, angle=0, params={}, shift_point=False, inverse=False, longlat=True):
    """Transforms a polygon from longlat to meter with respect to a shift point. For back transformation, a
    params['zone'] parameter must be supplied that is calculated in the first transformation
    """
    initial_polygon = np.array(initial_polygon)
    if len(initial_polygon) != 2:
        initial_polygon = initial_polygon.T
    if longlat:
        if not inverse:
            x_meter, y_meter, shift_point_out = transform_coordinates(initial_polygon[0], initial_polygon[1], angle, params, shift_point)
            polygon = np.array(list(zip(x_meter, y_meter)))
        else:
            x_long, y_lat = transform_coordinates(initial_polygon[0], initial_polygon[1], angle, params, shift_point, inverse=True)
            polygon = np.array(list(zip(x_long, y_lat)))
            shift_point_out = False
    else:
        polygon = np.array(list(zip(initial_polygon[0], initial_polygon[1])))
        shift_point_out = False
    return polygon, shift_point_out


def distance_space(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def plot_polygons(polygons):
    """Plots a list of polygons"""
    plt.figure()
    for polygon_i in polygons:
        polygon = polygon_i[:]
        polygon, _ = transform_polygon(polygon, angle=0, params={}, shift_point=0, inverse=False, longlat=False)
        if not all(polygon[0]==polygon[-1]):
            polygon = np.append(polygon, [polygon[0]], axis=0)
        plt.plot(polygon.T[0], polygon.T[1])
        plt.plot(polygon.T[0], polygon.T[1],'rx')
        plt.plot(polygon.T[0][0], polygon.T[1][0],'bx')


def plot_parcels_dict(parcels_dict, newplot=True, all_keys=False):
    """Plots a list of dictionaries with polygons at dict['geometry']['coordinates'][0]"""
    if newplot:
        plt.figure()
    for shape in parcels_dict:
        polygon = np.array(shape['geometry']['coordinates'][0]).T
        plt.plot(polygon[0], polygon[1])
        # plt.plot(polygon[0], polygon[1], 'r')
        label = ''
        if all_keys:
            for key in shape['properties'].keys():
                label += str(key) + ': ' + str(shape['properties'][key]) + '\n'
        else:
            if 'idp' in shape['properties'].keys():
                label += 'idp: ' + str(shape['properties']['idp']) + '\n'
            if 'name' in shape['properties'].keys():
                label += 'name: ' + str(shape['properties']['name']) + '\n'
        plt.text(np.min(polygon[0]), np.mean(polygon[1]), label)


def polygon_to_hectare(long_lat_polygon):
    """returns are in hectar for a polygon in long, lat coords"""
    meter_polygon, _ = transform_polygon(long_lat_polygon)
    area = Polygon(meter_polygon).area / 10000
    return area


import numpy as np
from metaurban.constants import MetaUrbanType, PGDrivableAreaProperty

def construct_lane(lane, lane_index):
    """
    Construct a physics body for the lane localization
    """
    if lane_index is not None:
        lane.index = lane_index
    # build physics contact
    if not lane.need_lane_localization:
        return []
    assert lane.polygon is not None, "Polygon is required for building lane"
    polygons = [lane.polygon]
    return polygons

def construct_continuous_line_polygon(lane, lateral, line_color, line_type):
    segment_num = int(lane.length / PGDrivableAreaProperty.LANE_SEGMENT_LENGTH)
    polygon = []
    if segment_num == 0:
        segment_polygon = []
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(i, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        if lane.length > int(lane.length):
            segment_polygon.append([lane.position(lane.length, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            segment_polygon.append([lane.position(lane.length, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(int(lane.length) - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
        polygon.append(np.array(segment_polygon).reshape(-1, 2))
    for segment in range(segment_num):
        if segment == segment_num - 1:
            segment_polygon = []
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            if lane.length > int(lane.length):
                segment_polygon.append([lane.position(lane.length, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
                segment_polygon.append([lane.position(lane.length, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(int(lane.length) - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        else:
            segment_polygon = []
            for i in range(int(PGDrivableAreaProperty.LANE_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            for i in range(int(PGDrivableAreaProperty.LANE_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position((segment + 1) * PGDrivableAreaProperty.LANE_SEGMENT_LENGTH - i, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        
    return polygon

def construct_continuous_polygon(lane, start_lat, end_lat):
    segment_num = int(lane.length / PGDrivableAreaProperty.SI_SEGMENT_LENGTH)
    polygon = []
    if segment_num == 0:
        segment_polygon = []
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(i, start_lat)])
        if lane.length > int(lane.length):
            segment_polygon.append([lane.position(lane.length, start_lat)])
            segment_polygon.append([lane.position(lane.length, end_lat)])
        for i in range(int(lane.length)):
            segment_polygon.append([lane.position(int(lane.length) - i, end_lat)])
        polygon.append(np.array(segment_polygon).reshape(-1, 2))
    for segment in range(segment_num):
        if segment == segment_num - 1:
            segment_polygon = []
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH, start_lat)])
            if lane.length > int(lane.length):
                segment_polygon.append([lane.position(lane.length, start_lat)])
                segment_polygon.append([lane.position(lane.length, end_lat)])
            for i in range(int(lane.length) - segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH + 1):
                segment_polygon.append([lane.position(int(lane.length) - i, end_lat)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        else:
            segment_polygon = []
            for i in range(int(PGDrivableAreaProperty.SI_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position(i + segment * PGDrivableAreaProperty.SI_SEGMENT_LENGTH, start_lat)])
            for i in range(int(PGDrivableAreaProperty.SI_SEGMENT_LENGTH) + 1):
                segment_polygon.append([lane.position((segment + 1) * PGDrivableAreaProperty.SI_SEGMENT_LENGTH - i, end_lat)])
            polygon.append(np.array(segment_polygon).reshape(-1, 2))
        
    return polygon

def construct_broken_line_polygon(lane, lateral, line_color, line_type):
    assert MetaUrbanType.is_broken_line(line_type)
    points = lane.get_polyline(2, lateral)
    polygon = []
    for index in range(0, len(points) - 1, 2):
        segment_polygon = []
        if index + 1 < len(points):
              segment_polygon.append([lane.position(2 * index, lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * (index + 1), lateral - PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * (index + 1), lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              segment_polygon.append([lane.position(2 * index, lateral + PGDrivableAreaProperty.LANE_LINE_WIDTH / 2)])
              polygon.append(np.array(segment_polygon).reshape(-1, 2))
    return polygon
import trimesh
from shapely.geometry import Polygon
def generate_random_road(area_size=(100, 100)):
    start = (0.25, 0.25)
    end = (np.random.uniform(4 * area_size[0] / 4, area_size[0]), np.random.uniform(4 * area_size[1] / 4, area_size[1]))
    
    mid_points_x = np.linspace(start[0], end[0], 9)
    mid_points_y = np.linspace(start[1], end[1], 9)
    mid_points_y[1:-1] = np.random.uniform(0, area_size[1], size=7)
    # mid_points_x[1] = 2.
    # mid_points_y[1] = 0.25

    x_points = mid_points_x
    y_points = mid_points_y
    
    from scipy.interpolate import make_interp_spline
    x_new = np.linspace(start[0], end[0], 300)
    spl = make_interp_spline(x_points, y_points, k=1)
    y_smooth = spl(x_new)
    
    return x_new, y_smooth

def get_road_trimesh(x, y, area_size, min_road_width=15, max_road_width=18, height=2.1, boundary=[-1e10, 1e10]):
    road_width = np.random.uniform(min_road_width, max_road_width)
    
    upper_boundary = y + road_width / 2
    lower_boundary = y - road_width / 2
    upper_boundary = upper_boundary.clip(boundary[0], boundary[1])
    lower_boundary = lower_boundary.clip(boundary[0], boundary[1])
    polyline_points = np.concatenate([np.column_stack((x, upper_boundary)), np.column_stack((x[::-1], lower_boundary[::-1]))])
    
    polygon = Polygon(polyline_points)
    
    mesh = trimesh.creation.extrude_polygon(polygon, height=height)
    
    return mesh, [upper_boundary, lower_boundary], polyline_points
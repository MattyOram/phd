import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
from phd_helpers.helpers import transform_points

def get_mc1_line(centre, dir, length):
    point1 = centre - dir * length / 2
    point2 = centre + dir* length / 2
    return pv.Line(point1, point2)

def get_axis_dir(mesh_points):
    pca = PCA(n_components=3)
    pca.fit(mesh_points)

    # first pc is main axis
    return pca.components_[0]

def project_to_plane_2d(point, origin, u, v, normal):
    # Vector from plane origin to point
    vec = point - origin
    
    # Remove component along the normal
    projected_vec = vec - np.dot(vec, normal) * normal
    
    # 2D coordinates in plane basis
    x = np.dot(projected_vec, u)
    y = np.dot(projected_vec, v)
    
    return np.array([x, y]).ravel()

def get_2d_basis(normal):
    normal /= np.linalg.norm(normal)
    cross_dir = -np.array([1, 0, 0]) if abs(normal[0]) < 0.95 else np.array([0, 1, 0])

    u = np.cross(normal, cross_dir)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def get_central_lines_2d(mesh_points, R1, t1, R2, t2, line_length=55):
    """central axis of mc1 in 2 poses and best fit plane and intersect - mush be in MC1 coordinate system"""
    # direction of central axis in mc1 coordinate system
    #axis_dir = get_axis_dir(mesh_points)
    axis_dir = np.array([1, 0, 0])
    line1_dir = transform_points(axis_dir, R1, np.zeros(3))
    line2_dir = transform_points(axis_dir, R2, np.zeros(3))

    # plane based on best fit to central axis points #
    # get points on each centre axis
    p1a = t1 + line1_dir * line_length/2 ############ if using coordinate system where the origin is not at the mc1 centroid -->
    p1b = t1 - line1_dir * line_length/2 ############ --> need to change origin of points , p, t1/2 + ... assumes translation from origin.
    p2a = t2 + line2_dir * line_length/2
    p2b = t2 - line2_dir * line_length/2
    points = np.vstack([p1a, p1b, p2a, p2b])

    # Fit plane 
    pca = PCA(n_components=2)
    pca.fit(points)
    plane_normal = np.cross(pca.components_[0], pca.components_[1])
    plane_centroid = points.mean(axis=0)
    
    # 2D basis
    plane_normal /= np.linalg.norm(plane_normal)
    u, v = get_2d_basis(plane_normal)

    # Project flexion axis
    line1_2d = np.vstack([
        project_to_plane_2d(p1a, plane_centroid, u, v, plane_normal),
        project_to_plane_2d(p1b, plane_centroid, u, v, plane_normal)
    ])

    # Project extension axix
    line2_2d = np.vstack([
        project_to_plane_2d(p2a, plane_centroid, u, v, plane_normal),
        project_to_plane_2d(p2b, plane_centroid, u, v, plane_normal)
    ])

    # intersection 
    line1_vec = line1_2d[0] - line1_2d[1]
    line2_vec = line2_2d[0] - line2_2d[1]
    m1 = line1_vec[1] / line1_vec[0]
    m2 = line2_vec[1] / line2_vec[0]

    c1 = line1_2d[0, 1] - m1*line1_2d[0, 0]
    c2 = line2_2d[0, 1] - m2*line2_2d[0, 0]

    x_int = (c2 - c1) / (m1 - m2)
    y_int = m1*x_int + c1

    int_2d = np.array([x_int, y_int])
    int_3d = plane_centroid + int_2d[0] * u + int_2d[1] * v



    return line1_2d, line2_2d, int_2d, int_3d, plane_normal, plane_centroid, line1_dir, line2_dir

def angle_2d(line1_2d, line2_2d):
    """input 2x2 (x, y) array of points for each line. Returns samllest angle"""
    line1_vec = line1_2d[0] - line1_2d[1]
    line2_vec = line2_2d[0] - line2_2d[1]

    line1_vec /= np.linalg.norm(line1_vec)
    line2_vec /= np.linalg.norm(line2_vec)

    dot_product = np.clip(np.dot(line1_vec, line2_vec), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot_product))
    return angle
import math

def angle_between_vectors(x, y):
    dot_product = sum([xi * yi for xi, yi in zip(x, y)])
    magnitude_x = math.sqrt(sum([xi ** 2 for xi in x]))
    magnitude_y = math.sqrt(sum([yi ** 2 for yi in y]))
    return math.degrees(math.acos(dot_product / (magnitude_x * magnitude_y)))

def check_vec_equ(landmark, p1, p2, p3, p4):
    x1, y1, z1 = landmark[p1].x, landmark[p1].y, landmark[p1].z
    x2, y2, z2 = landmark[p2].x, landmark[p2].y, landmark[p2].z
    x3, y3, z3 = landmark[p3].x, landmark[p3].y, landmark[p3].z
    x4, y4, z4 = landmark[p4].x, landmark[p4].y, landmark[p4].z
    
    v1 = [x2 - x1, y2 - y1, z2 - z1]
    v2 = [x4 - x3, y4 - y3, z4 - z3]
    
    return angle_between_vectors(v1, v2)

def check_point_dist(landmark, p1, p2):
    x1, y1, z1 = landmark[p1].x, landmark[p1].y, landmark[p1].z
    x2, y2, z2 = landmark[p2].x, landmark[p2].y, landmark[p2].z
    dist_line = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2) ** 0.5
    return dist_line

def get_hand_bbox(landmark, img_width, img_height):
    x = [lm.x * img_width for lm in landmark]
    y = [lm.y * img_height for lm in landmark]
    x_min = max(0, min(x) - 1)
    x_max = min(img_width, max(x) + 1)
    y_min = max(0, min(y) - 1)
    y_max = min(img_height, max(y) + 1)
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def get_overlap_ratio(data, idx1, idx2, mp_data=None):
    # calculate the IoU of two bounding boxes
    box1 = data[idx1][:4]
    box2 = data[idx2][:4]
    if mp_data:
        box2 = mp_data

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    return inter_area / area2

def extend_bbox(bbox, extend_ratio, img_width, img_height):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, x1 - w * extend_ratio)
    y1 = max(0, y1 - h * extend_ratio)
    x2 = min(img_width, x2 + w * extend_ratio)
    y2 = min(img_height, y2 + h * extend_ratio)
    return x1, y1, x2, y2
    

    
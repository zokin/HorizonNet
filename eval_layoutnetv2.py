import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

from eval_cuboid import prepare_gtdt_pairs
from dataset import cor_2_1d
from misc import post_proc

from scipy.spatial import HalfspaceIntersection, ConvexHull

def coorx2u(x, w=1024):
    return ((x + 0.5) / w - 0.5) * 2 * np.pi

def coory2v(y, h=512):
    return ((y + 0.5) / h - 0.5) * np.pi

def u2coorx(u, w=1024):
    return (u / (2 * np.pi) + 0.5) * w - 0.5

def v2coory(v, h=512):
    return (v / np.pi + 0.5) * h - 0.5

def uv2xy(u, v, z=-50):
    c = z / np.tan(v)
    x = c * np.cos(u)
    y = c * np.sin(u)
    return x, y

def pano_connect_points(p1, p2, z=-50, w=1024, h=512):
    u1 = coorx2u(p1[0], w)
    v1 = coory2v(p1[1], h)
    u2 = coorx2u(p2[0], w)
    v2 = coory2v(p2[1], h)

    x1, y1 = uv2xy(u1, v1, z)
    x2, y2 = uv2xy(u2, v2, z)

    if abs(p1[0] - p2[0]) < w / 2:
        pstart = np.ceil(min(p1[0], p2[0]))
        pend = np.floor(max(p1[0], p2[0]))
    else:
        pstart = np.ceil(max(p1[0], p2[0]))
        pend = np.floor(min(p1[0], p2[0]) + w)
    coorxs = (np.arange(pstart, pend + 1) % w).astype(np.float64)
    vx = x2 - x1
    vy = y2 - y1
    us = coorx2u(coorxs, w)
    ps = (np.tan(us) * x1 - y1) / (vy - np.tan(us) * vx)
    cs = np.sqrt((x1 + ps * vx) ** 2 + (y1 + ps * vy) ** 2)
    vs = np.arctan2(z, cs)
    coorys = v2coory(vs)

    return np.stack([coorxs, coorys], axis=-1)


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xy(coor, z=50, coorW=1024, coorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u)
    y = -c * np.cos(u)
    return np.hstack([x[:, None], y[:, None]])


def tri2halfspace(pa, pb, p):
    v1 = pa - p
    v2 = pb - p
    vn = np.cross(v1, v2)
    if -vn @ p > 0:
        vn = -vn
    return [*vn, -vn @ p]


def xyzlst2halfspaces(xyz_floor, xyz_ceil):
    '''
    return halfspace enclose (0, 0, 0)
    '''
    N = xyz_floor.shape[0]
    halfspaces = []
    for i in range(N):
        last_i = (i - 1 + N) % N
        next_i = (i + 1) % N

        p_floor_a = xyz_floor[last_i]
        p_floor_b = xyz_floor[next_i]
        p_floor = xyz_floor[i]
        p_ceil_a = xyz_ceil[last_i]
        p_ceil_b = xyz_ceil[next_i]
        p_ceil = xyz_ceil[i]
        halfspaces.append(tri2halfspace(p_floor_a, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_floor_a, p_ceil, p_floor))
        halfspaces.append(tri2halfspace(p_ceil, p_floor_b, p_floor))
        halfspaces.append(tri2halfspace(p_ceil_a, p_ceil_b, p_ceil))
        halfspaces.append(tri2halfspace(p_ceil_a, p_floor, p_ceil))
        halfspaces.append(tri2halfspace(p_floor, p_ceil_b, p_ceil))
    return np.array(halfspaces)

def layout_2_depth(cor_id, h, w, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    vc, vf = cor_2_1d(cor_id, h, w)
    vc = vc[None, :]  # [1, w]
    vf = vf[None, :]  # [1, w]
    assert (vc > 0).sum() == 0
    assert (vf < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    vs = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    vs = np.repeat(vs[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    floor_h = 1.6
    floor_d = np.abs(floor_h / np.sin(vs))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_h / np.tan(vf)

    # Ceiling-plane to depth
    ceil_h = np.abs(cs * np.tan(vc))      # [1, w]
    ceil_d = np.abs(ceil_h / np.sin(vs))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(vs))  # [h, w]

    # Recover layout depth
    floor_mask = (vs > vf)
    ceil_mask = (vs < vc)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)    # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth

def eval_3diou(dt_floor_coor, dt_ceil_coor, gt_floor_coor, gt_ceil_coor,
               ch=-1.6, coorW=1024, coorH=512):
    '''
    Evaluate 3D IoU of "convex layout".
    Instead of voxelization, this function use halfspace intersection
    to evaluate the volume.
    Input parameters:
        dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor
    have to be in shape [N, 2] and in the format of:
        [[x, y], ...]
    listing the corner position from left to right on the equirect image.
    '''
    dt_floor_coor = np.array(dt_floor_coor)
    dt_ceil_coor = np.array(dt_ceil_coor)
    gt_floor_coor = np.array(gt_floor_coor)
    gt_ceil_coor = np.array(gt_ceil_coor)
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0
    N = len(dt_floor_coor)
    dt_floor_xyz = np.hstack([
        np_coor2xy(dt_floor_coor, ch, coorW, coorH),
        np.zeros((N, 1)) + ch,
    ])
    gt_floor_xyz = np.hstack([
        np_coor2xy(gt_floor_coor, ch, coorW, coorH),
        np.zeros((N, 1)) + ch,
    ])
    dt_c = np.sqrt((dt_floor_xyz[:, :2] ** 2).sum(1))
    gt_c = np.sqrt((gt_floor_xyz[:, :2] ** 2).sum(1))
    dt_v2 = np_coory2v(dt_ceil_coor[:, 1], coorH)
    gt_v2 = np_coory2v(gt_ceil_coor[:, 1], coorH)
    dt_ceil_z = dt_c * np.tan(dt_v2)
    gt_ceil_z = gt_c * np.tan(gt_v2)

    dt_ceil_xyz = dt_floor_xyz.copy()
    dt_ceil_xyz[:, 2] = dt_ceil_z
    gt_ceil_xyz = gt_floor_xyz.copy()
    gt_ceil_xyz[:, 2] = gt_ceil_z

    dt_halfspaces = xyzlst2halfspaces(dt_floor_xyz, dt_ceil_xyz)
    gt_halfspaces = xyzlst2halfspaces(gt_floor_xyz, gt_ceil_xyz)

    in_halfspaces = HalfspaceIntersection(np.concatenate([dt_halfspaces, gt_halfspaces]),
                                          np.zeros(3))
    dt_halfspaces = HalfspaceIntersection(dt_halfspaces, np.zeros(3))
    gt_halfspaces = HalfspaceIntersection(gt_halfspaces, np.zeros(3))

    in_volume = ConvexHull(in_halfspaces.intersections).volume
    dt_volume = ConvexHull(dt_halfspaces.intersections).volume
    gt_volume = ConvexHull(gt_halfspaces.intersections).volume
    un_volume = dt_volume + gt_volume - in_volume

    return in_volume / un_volume

def eval_PE(dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor, H=512, W=1024):
    '''
    Evaluate pixel surface error (3 labels: ceiling, wall, floor)
    Input parameters:
        dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor
    have to be in shape [N, 2] and in the format of:
        [[x, y], ...]
    listing the corner position from left to right on the equirect image.
    '''
    y0 = np.zeros(W)
    y1 = np.zeros(W)
    y0_gt = np.zeros(W)
    y1_gt = np.zeros(W)
    # for j in range(dt_ceil_coor.shape[0]):
    for j in range(min(dt_ceil_coor.shape[0], gt_ceil_coor.shape[0])):
        coorxy = pano_connect_points(dt_ceil_coor[j], dt_ceil_coor[(j+1)%4], -50)
        y0[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(dt_floor_coor[j], dt_floor_coor[(j+1)%4], 50)
        y1[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_ceil_coor[j], gt_ceil_coor[(j+1)%4], -50)
        y0_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

        coorxy = pano_connect_points(gt_floor_coor[j], gt_floor_coor[(j+1)%4], 50)
        y1_gt[np.round(coorxy[:, 0]).astype(int)] = coorxy[:, 1]

    surface = np.zeros((H, W), dtype=np.int32)
    surface[np.round(y0).astype(int), np.arange(W)] = 1
    surface[np.round(y1).astype(int), np.arange(W)] = 1
    surface = np.cumsum(surface, axis=0)
    surface_gt = np.zeros((H, W), dtype=np.int32)
    surface_gt[np.round(y0_gt).astype(int), np.arange(W)] = 1
    surface_gt[np.round(y1_gt).astype(int), np.arange(W)] = 1
    surface_gt = np.cumsum(surface_gt, axis=0)

    return (surface != surface_gt).sum() / (H * W), surface, surface_gt

def test_general(dt_cor_id, gt_cor_id, w, h, losses):
    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)
    if not gt_poly.is_valid:
        print('Skip ground truth invalid (%s)' % gt_path)
        return

    # 2D IoU
    try:
        area_dt = dt_poly.area
        area_gt = gt_poly.area
        area_inter = dt_poly.intersection(gt_poly).area
        iou2d = area_inter / (area_gt + area_dt - area_inter)
    except:
        iou2d = 0

    # 3D IoU
    try:
        # cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
        # cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)
        # h_dt = abs(cch_dt.mean() - ch)
        # h_gt = abs(cch_gt.mean() - ch)
        # area3d_inter = area_inter * min(h_dt, h_gt)
        # area3d_pred = area_dt * h_dt
        # area3d_gt = area_gt * h_gt
        # iou3d = area3d_inter / (area3d_pred + area3d_gt - area3d_inter)
        iou3d = eval_3diou(dt_floor_coor, dt_ceil_coor, gt_floor_coor, 
            gt_ceil_coor)
    except:
        iou3d = 0

    # PE
    error, surface, surface_gt = eval_PE(dt_ceil_coor, dt_floor_coor, gt_ceil_coor, gt_floor_coor)
    pe = error

    # rmse & delta_1
    gt_layout_depth = layout_2_depth(gt_cor_id, h, w)
    try:
        dt_layout_depth = layout_2_depth(dt_cor_id, h, w)
    except:
        dt_layout_depth = np.zeros_like(gt_layout_depth)
    rmse = ((gt_layout_depth - dt_layout_depth)**2).mean() ** 0.5
    thres = np.maximum(gt_layout_depth/dt_layout_depth, dt_layout_depth/gt_layout_depth)
    delta_1 = (thres < 1.25).mean()

    # Add a result
    n_corners = len(gt_floor_coor)
    if n_corners % 2 == 1:
        n_corners = 'odd'
    elif n_corners < 10:
        n_corners = str(n_corners)
    else:
        n_corners = '10+'
    losses[n_corners]['2DIoU'].append(iou2d)
    losses[n_corners]['3DIoU'].append(iou3d)
    losses[n_corners]['rmse'].append(rmse)
    losses[n_corners]['delta_1'].append(delta_1)
    losses[n_corners]['PE'].append(pe)
    losses['overall']['2DIoU'].append(iou2d)
    losses['overall']['3DIoU'].append(iou3d)
    losses['overall']['rmse'].append(rmse)
    losses['overall']['delta_1'].append(delta_1)
    losses['overall']['PE'].append(pe)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dt_glob',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be json from inference.py')
    parser.add_argument('--gt_glob',
                        help='NOTE: Remeber to quote your glob path.'
                             'Files assumed to be txt')
    parser.add_argument('--w', default=1024, type=int,
                        help='GT images width')
    parser.add_argument('--h', default=512, type=int,
                        help='GT images height')
    args = parser.parse_args()

    # Prepare (gt, dt) pairs
    gtdt_pairs = prepare_gtdt_pairs(args.gt_glob, args.dt_glob)

    # Testing
    losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': [], 'PE': []})
        for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
    ])
    for gt_path, dt_path in tqdm(gtdt_pairs, desc='Testing'):
        # Parse ground truth
        with open(gt_path) as f:
            gt_cor_id = np.array([l.split() for l in f], np.float32)

        # Parse inferenced result
        with open(dt_path) as f:
            dt = json.load(f)
        dt_cor_id = np.array(dt['uv'], np.float32)
        dt_cor_id[:, 0] *= args.w
        dt_cor_id[:, 1] *= args.h

        test_general(dt_cor_id, gt_cor_id, args.w, args.h, losses)

    for k, result in losses.items():
        iou2d = np.array(result['2DIoU'])
        iou3d = np.array(result['3DIoU'])
        rmse = np.array(result['rmse'])
        delta_1 = np.array(result['delta_1'])
        pe = np.array(result['PE'])
        if len(iou2d) == 0:
            continue
        print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
        print('    2DIoU  : %.2f' % (iou2d.mean() * 100))
        print('    3DIoU  : %.2f' % (iou3d.mean() * 100))
        print('    RMSE   : %.2f' % (rmse.mean()))
        print('    delta^1: %.2f' % (delta_1.mean()))
        print('    PE     : %.2f' % (pe.mean(axis=0)))

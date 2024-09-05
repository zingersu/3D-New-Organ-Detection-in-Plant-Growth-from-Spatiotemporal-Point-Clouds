# -*- coding: utf-8 -*-

import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split('_')[1])
    return Initial


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def _call__(self, pts, k):
        farthest_pts = np.zeros((k, 5),dtype=np.float32)
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0, :3], pts[:, :3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self._calc_distances(farthest_pts[i, :3], pts[:, :3]))
        return farthest_pts


if __name__ == '__main__':
    path = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_txt")
    saved_path = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps")

    if os.path.exists(saved_path) == False:
        os.makedirs(saved_path)

    Filelist = sorted(os.listdir(path), key=sort_by_number)
    n = len(Filelist)
    for idx in range(n):
        points = np.loadtxt(os.path.join(path, Filelist[idx]))
        txt_array = np.array(points)
        print(Filelist[idx])
        sample_count = 2048

        FPS = FarthestSampler()
        sample_points = FPS._call__(txt_array, sample_count)

        file_nameR = os.path.splitext(Filelist[idx])[0] + "_fps" + ".txt"

        np.savetxt(os.path.join(saved_path, file_nameR), sample_points, fmt='%f %f %f %d %d')
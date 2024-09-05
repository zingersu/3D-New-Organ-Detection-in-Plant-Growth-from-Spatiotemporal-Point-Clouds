from __future__ import division
from __future__ import print_function

import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

def sort_by_number(file_name):
    Initial = int(file_name.split('_')[1])
    return Initial


all_folder_norm = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps")
output_folder = os.path.join(ROOT_DIR, "backbone_network\\data\\norm_data_fps_ICP")

if os.path.exists(output_folder) == False:
    os.makedirs(output_folder)

all_files_norm = sorted(os.listdir(all_folder_norm),key = sort_by_number)
file_numbers = len(all_files_norm)

for i in range(file_numbers - 1):
    source_file_norm = all_files_norm[i]
    target_file_norm = all_files_norm[i + 1]
    parts_to_check = [source_file_norm.split('_')[2] == target_file_norm.split('_')[2],
                      source_file_norm.split('_')[3] == target_file_norm.split('_')[3],
                      source_file_norm.split('_')[4] == target_file_norm.split('_')[4]]
    if all(parts_to_check):

        source_path = all_folder_norm + '\\' + source_file_norm
        target_path = all_folder_norm + '\\' + target_file_norm

        fish_target = np.loadtxt(target_path)
        A = fish_target[:, :3]
        target_label = fish_target[:, -2:]
        N = A.shape[0]
        dim = A.shape[1]

        fish_source = np.loadtxt(source_path)
        B = fish_source[:, :3]
        source_label = fish_source[:, -2:]

        all_label = np.vstack((source_label, target_label))

        num_tests = 100
        noise_sigma = .01


        def test_best_fit():


            total_time = 0

            for i in range(num_tests):


                start = time.time()
                T, R1, t1 = best_fit_transform(A, B)
                total_time += time.time() - start

            return


        def plot_registration(A, B, ax, iteration=1):
            ax.clear()

            ax.scatter(A[:, 0], A[:, 1], A[:, 2], c='blue', label='Target')
            ax.scatter(B[:, 0], B[:, 1], B[:, 2], c='red', label='Source')
            ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
                iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
            ax.legend(loc='upper left', fontsize='x-large')
            plt.draw()
            plt.pause(1)


        def best_fit_transform(A, B):

            assert A.shape == B.shape

            # get number of dimensions
            m = A.shape[1]

            # translate points to their centroids
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - centroid_A
            BB = B - centroid_B

            H = np.dot(AA.T, BB)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)

            # special reflection case
            if np.linalg.det(R) < 0:
                Vt[m - 1, :] *= -1
                R = np.dot(Vt.T, U.T)

            t = centroid_B.T - np.dot(R, centroid_A.T)

            T = np.identity(m + 1)
            T[:m, :m] = R
            T[:m, m] = t

            return T, R, t


        def nearest_neighbor(src, dst):

            assert src.shape == dst.shape

            neigh = NearestNeighbors(n_neighbors=1)
            neigh.fit(dst)
            distances, indices = neigh.kneighbors(src, return_distance=True)

            return distances.ravel(), indices.ravel()


        def icp(B, A, init_pose=None, max_iterations=50, tolerance=0.00001):

            assert A.shape == B.shape

            # get number of dimensions
            m = A.shape[1]

            # make points homogeneous, copy them to maintain the originals
            src = np.ones((m + 1, A.shape[0]))
            dst = np.ones((m + 1, B.shape[0]))
            src[:m, :] = np.copy(A.T)
            dst[:m, :] = np.copy(B.T)

            # apply the initial pose estimation
            if init_pose is not None:
                src = np.dot(init_pose, src)

            prev_error = 0

            for i in range(max_iterations):
                distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

                T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

                src = np.dot(T, src)

                # check error
                mean_error = np.mean(distances)
                if np.abs(prev_error - mean_error) < tolerance:
                    break
                prev_error = mean_error

            # calculate final transformation
            T, _, _ = best_fit_transform(A, src[:m, :].T)

            return T, distances, i


        def test_icp():

            total_time = 0
            min_avg_distance = float('inf')
            best_T = None

            for i in range(num_tests):

                # Run ICP
                start = time.time()
                T, distances, iterations = icp(B, A, tolerance=0.000001)

                avg_distance = np.mean(distances)
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_T = T

                total_time += time.time() - start


            print('最小平均距离: {:.8}'.format(min_avg_distance))
            print('对应的最佳齐次矩阵：')
            print(best_T)

            return best_T


        if __name__ == "__main__":
            test_best_fit()
            best_T = test_icp()

            C = np.ones((N, 4))
            C[:, 0:3] = A

            D = np.ones((N, 4))
            D[:, 0:3] = B

            E = np.dot(best_T, C.T).T

            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # plot_registration(A, B, ax=ax)
            # plot_registration(E[:, :3], B, ax=ax)
            # plt.show()

            all_point = np.vstack((D, E))
            all_point = np.hstack((all_point[:, :3], all_label))

            save_name = output_folder + "\\" + "_".join(source_file_norm.split("_")[:-1]) + " & " + target_file_norm.split("_")[-2] + ".txt"
            np.savetxt(save_name , all_point, fmt="%.6f %.6f %.6f %d %d", delimiter=" ")

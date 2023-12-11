import numpy as np
def classify_points(dist_matrix, threshold=1):
    n = len(dist_matrix)
    classes = [-1] * n
    class_idx = 0
    for i in range(n):
        if classes[i] != -1:
            continue
        classes[i] = class_idx
        for j in range(i + 1, n):
            if max(dist_matrix[i][j],dist_matrix[j][i]) < threshold:
                classes[j] = class_idx
        class_idx += 1
    return classes

def find_redundant_positions(lst):
    last_positions = {}
    for i, x in enumerate(lst):
        last_positions[x] = i

    minred_actions_list = [i for i, x in enumerate(lst) if i in last_positions.values()]
    redundant_actions_list = [i for i, x in enumerate(lst) if i not in last_positions.values()]
    return minred_actions_list, redundant_actions_list



if __name__ == '__main__':
    kl_path = "D:\\research\\ar_robotics_remotelog\\0317测试新mask\\Mask_unlockpickupar-v0_Nill_PPO_n-1_2023-03-16-17-18-18\\kl_keydoor.txt"
    test_kl = np.loadtxt(kl_path)
    # print(test_kl)
    classes = classify_points(test_kl)
    print(classes)
    minred_actions_list, redundant_actions_list = find_redundant_positions(classes)
    print(minred_actions_list)
    print(redundant_actions_list)
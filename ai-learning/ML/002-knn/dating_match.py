from numpy import zeros, array, shape, tile
import matplotlib
from matplotlib import pyplot


class DatingMatch(object):
    def __init__(self):
        pass

    def matrix_from_file(self, filename="../datasets/KNN/datingTestSet2.txt"):
        """
        Get matrix from file data.
        :param filename:
        :return:
        """
        file = open(filename)
        lines = file.readlines()
        # init matrix
        matrix = zeros((len(lines), 3))
        class_label_vector = list()

        index = 0
        for line in lines:
            values = line.strip().split("\t")
            matrix[index, :] = values[0:3]
            class_label_vector.append(int(values[-1]))
            index += 1

        return matrix, class_label_vector

    def draw_scatter(self, matrix, labels):
        """
        绘制散点图
        :param matrix:
        :param labels:
        :return:
        """
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.scatter(matrix[:, 0], matrix[:, 1], 10.0 * array(labels), 10.0 * array(labels))
        pyplot.show()

    def auto_norm(self, matrix):
        """
        归一化特征值，消除属性之间量级不同导致的影响。
        归一化公式：
            y = (x - x_min)/(x_max - x_min)
            其中，min和max分别是数据集中的最小特征值和最大特征值。该函数可自动将数字特征值转化为0~1的区间。
        :param matrix:
        :return:
        """
        min_vals = matrix.min(0)
        max_vals = matrix.max(0)

        ranges = max_vals - min_vals

        norm_matrix = zeros(shape(matrix))
        m = matrix.shape[0]
        norm_matrix = matrix - tile(min_vals, (m, 1))
        norm_matrix = norm_matrix / tile(ranges, (m, 1))

        # or
        # norm_matrix =

        return norm_matrix, ranges, min_vals

    def core_classify(self, in_x, matrix, labels, k):
        """

        :param in_x:
        :param matrix:
        :param labels:
        :param k:
        :return:
        """
        matrix_size = matrix.shape[0]

        # 计算欧氏距离
        diff_matrix = tile(in_x, (matrix_size, 1)) - matrix
        #
        square_diff_matrix = diff_matrix ** 2
        #
        square_distances = square_diff_matrix.sum(axis=1)
        distances = square_distances ** 0.5

        # 按distance排序后的索引数组
        vote_count = {}
        sorted_distance = distances.argsort()
        for i in range(0, k):
            vote_label = labels[sorted_distance[i]]
            vote_count[vote_label] = vote_count.get(vote_label, 0) + 1

        max_vote_count = max(vote_count, key=vote_count.get)
        return max_vote_count


if __name__ == "__main__":
    dm = DatingMatch()
    dm_matrix, dm_labels = dm.matrix_from_file()
    dm.draw_scatter(dm_matrix, dm_labels)
    # 归一化
    norm_matrix, ranges, min_vals = dm.auto_norm(dm_matrix)
    print(norm_matrix)
    print(ranges)
    print(min_vals)
    x = [0.0, 0.10, 0.11]
    k = 10
    belong = dm.core_classify(x, norm_matrix, dm_labels, k)
    print(belong)
from os import listdir
from numpy import tile, zeros


class HandwritingNumberRecognise(object):
    def __init__(self):
        pass

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

    def matrix_from_img(self, filename=''):
        """
        将图片转化为一维向量
        :param filename:
        :return:
        """
        vector = zeros((1, 1024))
        file = open(filename)
        for i in range(32):
            line = file.readline()
            for j in range(32):
                vector[0, 32 * i + j] = int(line[j])

        return vector

    def recognise(self):
        training_dataset_dir = "../datasets/KNN/trainingDigits"
        testing_dataset_dir = "../datasets/KNN/testDigits"

        supervisor_labels = list()
        training_files = listdir(training_dataset_dir)

        m = len(training_files)
        training_matrix = zeros((m, 1024))
        for idx, filename in enumerate(training_files):
            # parse number from filename
            filename_without_suffix = filename.split(".")[0]
            digit = int(filename_without_suffix.split("_")[0])
            #
            supervisor_labels.append(digit)
            training_matrix[idx, :] = self.matrix_from_img("{}/{}".format(training_dataset_dir, filename))
            # print(filename, filename_without_suffix, digit)

        testing_files = listdir(testing_dataset_dir)
        error_count = 0
        for idx, filename in enumerate(testing_files):
            # parse real number from filename
            filename_without_suffix = filename.split(".")[0]
            digit = int(filename_without_suffix.split("_")[0])

            vector = self.matrix_from_img("{}/{}".format(testing_dataset_dir, filename))

            classify_res = self.core_classify(vector, training_matrix, supervisor_labels, 3)
            print("分类结果：{}，实际值：{}".format(classify_res, digit))

            if classify_res != digit:
                print("【分类错误】文件名({})".format(filename))
                error_count += 1

        print(error_count)


if __name__ == "__main__":
    hwr = HandwritingNumberRecognise()
    # vector = hwr.matrix_from_img(filename="../datasets/KNN/trainingDigits/0_0.txt")
    # print(vector)
    hwr.recognise()
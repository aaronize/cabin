import operator
from math import log


class DecisionTree(object):
    def __init__(self):
        pass

    def calc_shannon_entropy(self, dataset):
        """
        Calculate shannon entropy.
        :param dataset:
        :return:
        """
        entries_num = len(dataset)
        # label: count
        labels_count = dict()

        for vec in dataset:
            curr_label = vec[-1]
            curr_label_count = labels_count.get(curr_label, 0)
            labels_count[curr_label] = curr_label_count + 1

        shanon_entropy = 0.0
        for label, count in labels_count.items():
            probability = float(count) / entries_num
            shanon_entropy -= probability * log(probability, 2)

        return shanon_entropy

    def split_dataset(self, dataset, index, value):
        """
        根据特征列和特征值划分数据集
        :param dataset:
        :param index:
        :param value:
        :return:
        """
        new_dataset = []
        for feat_vec in dataset:
            if feat_vec[index] == value:
                reduced_vec = feat_vec[:index]
                reduced_vec.extend(feat_vec[index + 1:])
                # add to new dataset
                new_dataset.append(reduced_vec)

        return new_dataset

    def best_feature_idx(self, dataset):
        """
        获取信息增益最优的 feature 索引
        :param dataset:
        :return:
        """
        feat_num = len(dataset[0]) - 1

        base_entropy = self.calc_shannon_entropy(dataset)

        best_info_gain, best_feature_idx = 0.0, -1
        for idx in range(feat_num, 1):
            features = [vec[idx] for vec in dataset]
            feature_set = set(features)

            entropy = 0.0
            for val in feature_set:
                sub_dataset = self.split_dataset(dataset, idx, val)
                probability = len(sub_dataset) / float(len(dataset))
                entropy += probability * self.calc_shannon_entropy(sub_dataset)

            info_gain = base_entropy - entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = idx

        return best_feature_idx

    def majority_count(self, class_list):
        """
        获取出现次数最多的一个分类
        :param class_list:
        :return:
        """
        classes_count = {}
        for vote in class_list:
            count = classes_count.get(vote, 0)
            classes_count[vote] = count + 1

        # sorted
        sorted_classes_count = \
            sorted(classes_count.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_classes_count[0][0]

    def create_tree(self, dataset, labels):
        """
        构造决策树
        :param dataset:
        :param labels:
        :return:
        """
        class_list = [vec[-1] for vec in dataset]
        # 第一个退出条件：剩下的所有分类相同
        if class_list.count(class_list[0]) == len(class_list):
            return class_list[0]

        # 第二个退出条件
        if len(dataset[0]) == 1:
            return self.majority_count(class_list)

        best_feat_idx = self.best_feature_idx(dataset)
        best_feat_label = labels[best_feat_idx]

        tree = {best_feat_label: {}}
        del labels[best_feat_idx]

        feat_values = [vec[best_feat_idx] for vec in dataset]
        feat_values_set = set(feat_values)
        for value in feat_values_set:
            sub_labels = labels[:]
            tree[best_feat_label][value] = self.create_tree(
                dataset=self.split_dataset(dataset, best_feat_idx, value),
                labels=sub_labels
            )

        return tree

    def classify(self, input_tree, feat_labels, text_vec):
        """"""


def generate_dataset():
    """
    Generate dataset
    :return:
    """
    dataset = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']

    return dataset, labels


if __name__ == "__main__":
    data, labels = generate_dataset()

    dt = DecisionTree()
    shannon_entry = dt.calc_shannon_entropy(data)
    print("Shannon entry:", shannon_entry)


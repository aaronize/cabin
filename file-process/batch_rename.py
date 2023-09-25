import os


def rename_batch(path=''):
    """
    重命名
    :param path:
    :return:
    """
    try:
        file_list = os.listdir(path)
    except Exception as e:
        print("请输入合法的路径！", e.args)
        return

    if not len(file_list):
        print("该路径下没有文件！")
        return

    print(f"开始批量修改{path}目录下的文件...")
    for file in file_list:
        old_fullname = os.path.join(path, file)
        split_name_parts = file.split('@')
        if len(split_name_parts) < 2:
            print(f"已跳过文件{file}")
            continue

        new_filename = split_name_parts[0] + '@' + split_name_parts[1] + '.pdf'
        new_fullname = os.path.join(path, new_filename)
        print(f"已{file}修改为：{new_filename}")
        os.rename(old_fullname, new_fullname)

    print("所有文件修改完毕！")


if __name__ == '__main__':
    path = input('请输入文件所在目录的全路径：')
    if not path:
        raise Exception('输入路径为空！')

    rename_batch(path)


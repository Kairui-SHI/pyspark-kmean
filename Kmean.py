import numpy as np
from pyspark.sql import Row
from pyspark.sql.functions import col, mean
from pyspark.ml.linalg import Vectors

def kmeans(data_rdd, k = 3, seed = None, epoch = 10):
    # 初始化聚类中心
    np.random.seed(seed)
    length = 1
    lb = 0
    up = 256
    centroids = np.random.randint(lb, up, size=(k, length))


    i = 0
    while i < epoch:
        print('i = %d' % i)
        # 需要将中心点按照大小排列
        centroids = np.sort(centroids.flatten()).reshape(centroids.shape)
        # 计算每个样本点到每个聚类中心的距离
        distances = data_rdd.map(lambda point: (np.argmin([np.linalg.norm(point - c) for c in centroids]), point))
        # np.linalg.norm(point - c) for c in centroids 从centroids中读取中心点，用每个特征值与其求差值，然后用norm函数求每个向量的
        # 长度，即每个点与各个中心点的欧式距离,通过argmin这个函数找到最小值的索引，再返回给rdd数据类型，这样就能得到点对类的归类
        # 利用map对rdd数据类型中的每个样本点进行操作，记每个样本点为point
        # 将每个点分配给最近的聚类中心

        i += 1
        if i < epoch:
            clusters = distances.groupByKey()
            # 通过groupByKey(对第一列元素group)对distances (cluster_index, point)分类，把同一类的点归类
            # 得到的cluster为rdd数据格式，并且有多少类就有多少行，每行第一个元素代表类号，第二个元素是一个迭代器代表该类别中的所有点

            # 计算新的聚类中心
            new_centroids_rdd = clusters.map(lambda cluster: np.round(np.mean(np.array(list(cluster[1])), axis=0)))
            new_centroids = new_centroids_rdd.collect()
            # 通过map读取clusters中每一类中所有点 np.array(list(cluster[1]))并转换成数组格式，
            # np.mean( ... , axis=0)) 代表计算每一列各自的均值，这里每一列代表每一类 均值即为新的类中心
            # 最后整数化返回给new_centroids

            # 更新聚类中心
            centroids = np.array(new_centroids)

    # 预测
    predictions = distances.map(lambda x: Row(features=Vectors.dense(x[1]), prediction=int(x[0])))
    print(centroids)

    return predictions



import findspark
findspark.init()

import numpy as np
from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import when, col, lit, monotonically_increasing_id as mi
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Extract_pic import bmp2tif, tif_rgb2gray, get_pixelvalue, produce_img
from pyspark.ml.evaluation import ClusteringEvaluator
from PIL import Image
import time

if __name__ == "__main__":
    start_time = time.time()

    # process image
    filepath = 'input.bmp'
    bmp2tif(filepath)
    image = tif_rgb2gray()
    get_pixelvalue(image)
    height = image.height
    width = image.width
    image.close()

    # build RDD
    spark = SparkContext(appName="KMeans_pyspark", master='local[16]')  # SparkContext
    SparkSession(spark)

    # convert data to feature vector
    def f(x):
        rel = {}
        rel['pixel_values'] = Vectors.dense(float(x[0]))
        return rel

    # read data
    data = spark.textFile('./pixel_values.txt').map(lambda line:line.split('\t')).map(lambda p: Row(**f(p))).toDF()
    data.show()

    k = 3

    kmean_model = KMeans(featuresCol='pixel_values').setK(k)
    kmean_run = kmean_model.fit(data)
    kmeans_results = kmean_run.transform(data)

    kmeans_results = kmeans_results.withColumn("new_pixel_value", lit(0))

    # 查看聚类中心
    cluster_centers = kmean_run.clusterCenters()
    print("Cluster Centers:")
    for center in cluster_centers:
        print(center)

    kmeans_results = kmeans_results.withColumnRenamed("pixel_values", "features")
    evaluator = ClusteringEvaluator()
    # 设置评估指标为'silhouette'
    Silhouette = evaluator.evaluate(kmeans_results)
    print("Silhouette Score:", Silhouette)
    kmeans_results = kmeans_results.withColumnRenamed("features", "pixel_values")

    new_pixel_value_list = []
    new_pixel_value_list.append(0.0)
    for i in range(1, k-1):
        new_pixel_value_list.append(i*(256.0/(k-1)))
    new_pixel_value_list.append(256.0)

    for cluster in range(0, k):
        new_pv = new_pixel_value_list[cluster]
        kmeans_results = kmeans_results.withColumn("new_pixel_value", when(col("prediction") == cluster, new_pv).otherwise(col("new_pixel_value")))

    pixels_data = kmeans_results.select("new_pixel_value").collect()
    pixels = [row.new_pixel_value for row in pixels_data]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"经过的时间：{elapsed_time} 秒")

    produce_img(pixels, width, height)
    new_image = Image.open('classify.png')
    new_image.show()
    new_image.close()

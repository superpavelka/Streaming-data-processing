# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, CountVectorizer, StringIndexer, IndexToString

spark = SparkSession.builder.appName("gogin_spark").getOrCreate()

df1 = spark.read.load("id_data_csv",
                      format="csv", sep=",", inferSchema="true", header="true")

df2 = spark.read.load("id_diag_csv",
                      format="csv", sep=",", inferSchema="true", header="true")

df1.createOrReplaceTempView("id_data")
df2.createOrReplaceTempView("id_diag")

spark.read.parquet("/apps/spark/warehouse/sint_sales.db/sales_known").createOrReplaceTempView("sales_known")
spark.read.parquet("/apps/spark/warehouse/sint_sales.db/users_known").createOrReplaceTempView("users_known")

spark.sql("select * from id_data").show(10, False)
spark.sql("select * from id_diag").show(10, False)

spark.sql("select * from sales_known").show(10, False)
spark.sql("select * from users_known").show(10, False)

patients_known = spark.sql("""
select *
from id_data 
join id_diag  
where id_data.id = id_diag.id """)

# считаем сегмент зависимым от количества покупок клиента, суммы всех купленых товаров клиента, максимального числа купленных товаров клиента, минимального числа купленных товаров клиента, суммы потраченных рублей клиента, максимально потраченных рублей клиента, минимально потраченных рублей клиента
users_known = spark.sql("""
select count(*) as c, sum(items_count) as s1, max(items_count) as ma1, min(items_count) as mi1,
sum(price) as s2, max(price) as ma2, min(price) as mi2 ,u.gender, u.age, u.user_id, u.segment 
from sales_known s join users_known u 
where s.user_id = u.user_id 
group by u.user_id, u.gender, u.age, u.segment""")

# подробное описание модели https://towardsdatascience.com/machine-learning-with-pyspark-and-mllib-solving-a-binary-classification-problem-96396065d2aa
# и https://spark.apache.org/docs/latest/ml-features.html
# в общем - все анализируемые колонки заносим в колонку-вектор features
categoricalColumns = ['gender', 'age']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index').setHandleInvalid("keep")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                     outputCols=[categoricalCol + "classVec"]).setHandleInvalid("keep")
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol='segment', outputCol='label').setHandleInvalid("keep")
stages += [label_stringIdx]

numericCols = ['c', 's1', 'ma1', 'mi1', 's2', 'ma2', 'mi2']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
stages += [lr]

label_stringIdx_fit = label_stringIdx.fit(users_known)
indexToStringEstimator = IndexToString().setInputCol("prediction").setOutputCol("category").setLabels(
    label_stringIdx_fit.labels)

stages += [indexToStringEstimator]

pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(users_known)

# сохраняем модель на HDFS
pipelineModel.write().overwrite().save("my_LR_model8")

###для наглядности
pipelineModel.transform(users_known).select("segment", "category").show(
    100)  # можно посчитать процент полной сходимости

categoricalColumns = []
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index').setHandleInvalid("keep")
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()],
                                     outputCols=[categoricalCol + "classVec"]).setHandleInvalid("keep")
    stages += [stringIndexer, encoder]

label_stringIdx = StringIndexer(inputCol='diagnosis', outputCol='label').setHandleInvalid("keep")
stages += [label_stringIdx]

numericCols = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
               'concavity_mean',
               'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
               'perimeter_se',
               'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
               'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
               'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
               'fractal_dimension_worst']

assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features").setHandleInvalid("keep")
stages += [assembler]

lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
stages += [lr]

label_stringIdx_fit = label_stringIdx.fit(patients_known)
indexToStringEstimator = IndexToString().setInputCol("prediction").setOutputCol("category").setLabels(
    label_stringIdx_fit.labels)

stages += [indexToStringEstimator]

pipeline = Pipeline().setStages(stages)
pipelineModel = pipeline.fit(patients_known)

# сохраняем модель на HDFS
pipelineModel.write().overwrite().save("my_LR_model_patients")

###для наглядности
pipelineModel.transform(patients_known).select("diagnosis", "category").show(
    100)  # можно посчитать процент полной сходимости

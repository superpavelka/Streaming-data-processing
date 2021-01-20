# export SPARK_KAFKA_VERSION=0.10
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, TimestampType, FloatType
from pyspark.sql import functions as F
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, CountVectorizer, StringIndexer, IndexToString

spark = SparkSession.builder.appName("my_spark").getOrCreate()
kafka_brokers = "bigdataanalytics-worker-0.novalocal:6667"

# загружаю свои данные
df1 = spark.read.load("id_data_csv",
                      format="csv", sep=",", inferSchema="true", header="true")
df2 = spark.read.load("id_diag_csv",
                      format="csv", sep=",", inferSchema="true", header="true")
# создаю для них представление
df1.createOrReplaceTempView("id_data")
df2.createOrReplaceTempView("id_diag")
# проверяю как данные загрузились
spark.sql("select * from id_data").show(10, False)
spark.sql("select * from id_diag").show(10, False)
# делаю join двух таблиц, чтобы обучить модель
patients_known = spark.sql("""
select *
from id_data 
join id_diag  
where id_data.id = id_diag.id """)
# обучение модели
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
# проверяем работу модели и выдаем столбцы диагноза из файла и предсказанный
pipelineModel.transform(patients_known).select("diagnosis", "category").show(100)

# создаю в кафке новый топик
# /usr/hdp/3.1.4.0-315/kafka/bin/kafka-topics.sh --create --topic patients_json2 --zookeeper bigdataanalytics-worker-0.novalocal:2181 --partitions 3 --replication-factor 2 --config retention.ms=-1
# копирую на hdfs данные для топика кафки
# hdfs dfs -put csv/id_data_unknown2.csv csv_stream/id_data_unknown2.csv

# описание данных
schema = StructType() \
    .add("id", IntegerType()) \
    .add("radius_mean", FloatType()) \
    .add("texture_mean", FloatType()) \
    .add("perimeter_mean", FloatType()) \
    .add("area_mean", FloatType()) \
    .add("smoothness_mean", FloatType()) \
    .add("compactness_mean", FloatType()) \
    .add("concavity_mean", FloatType()) \
    .add("concave points_mean", FloatType()) \
    .add("symmetry_mean", FloatType()) \
    .add("fractal_dimension_mean", FloatType()) \
    .add("radius_se", FloatType()) \
    .add("texture_se", FloatType()) \
    .add("perimeter_se", FloatType()) \
    .add("area_se", FloatType()) \
    .add("smoothness_se", FloatType()) \
    .add("compactness_se", FloatType()) \
    .add("concavity_se", FloatType()) \
    .add("concave points_se", FloatType()) \
    .add("symmetry_se", FloatType()) \
    .add("fractal_dimension_se", FloatType()) \
    .add("radius_worst", FloatType()) \
    .add("texture_worst", FloatType()) \
    .add("perimeter_worst", FloatType()) \
    .add("area_worst", FloatType()) \
    .add("smoothness_worst", FloatType()) \
    .add("compactness_worst", FloatType()) \
    .add("concavity_worst", FloatType()) \
    .add("concave points_worst", FloatType()) \
    .add("symmetry_worst", FloatType()) \
    .add("fractal_dimension_worst", FloatType())

# загружаю данные из файла в поток
raw_files = spark \
    .readStream \
    .format("csv") \
    .schema(schema) \
    .options(path="csv_stream", header=True) \
    .load()


# проверяю что записалось с помощью вывода в консоль
def console_output(df, freq):
    return df.writeStream \
        .format("console") \
        .trigger(processingTime='%s seconds' % freq) \
        .options(truncate=False) \
        .start()


out = console_output(raw_files, 5)
out.stop()


# записываю данные из потока в кафку и конвертирую csv в json
def kafka_sink_json(df, freq):
    return df.selectExpr("CAST(null AS STRING) as key", "CAST(to_json(struct(*)) AS STRING) as value") \
        .writeStream \
        .format("kafka") \
        .trigger(processingTime='%s seconds' % freq) \
        .option("topic", "patients_json2") \
        .option("kafka.bootstrap.servers", kafka_brokers) \
        .option("checkpointLocation", "my_kafka_checkpoint5") \
        .start()


stream = kafka_sink_json(raw_files, 5)
stream.stop()

patients = spark.readStream. \
    format("kafka"). \
    option("kafka.bootstrap.servers", kafka_brokers). \
    option("subscribe", "patients_json2"). \
    option("startingOffsets", "earliest"). \
    option("maxOffsetsPerTrigger", "1"). \
    load()
# парсинг данных из потока
value_patients = patients.select(F.from_json(F.col("value").cast("String"), schema).alias("value"), "offset")
patients_flat = value_patients.select(F.col("value.*"), "offset")

# проверяю спарсенные данные
s = console_output(patients_flat, 5)
s.stop()

# создаю в кассандре новый keyspace и таблицу
# /cassandra/bin/cqlsh 10.0.0.18

'''
CREATE  KEYSPACE  task8_course
  WITH REPLICATION = {
     'class' : 'SimpleStrategy', 'replication_factor' : 1 } ;

CREATE TABLE task8_course.patients
(id int, 
radius_mean float,
texture_mean float,	
perimeter_mean float,
area_mean float,
smoothness_mean float,
compactness_mean float,
concavity_mean float,
concave_points_mean float,
symmetry_mean float,
fractal_dimension_mean float,
radius_se float,
texture_se float,
perimeter_se float,
area_se float,
smoothness_se float,
compactness_se float,
concavity_se float,
concave_points_se float,
symmetry_se float,
fractal_dimension_se float,
radius_worst float,
texture_worst float,
perimeter_worst float,
area_worst float,
smoothness_worst float,
compactness_worst float,
concavity_worst float,
concave_points_worst float,
symmetry_worst float,
fractal_dimension_worst float,
primary key (id));
'''
# спарк с коннектором для кассандры
# /spark2.4/bin/pyspark --packages org.apache.spark:spark-sql-kafka-0-10_2.11:2.4.5,com.datastax.spark:spark-cassandra-connector_2.11:2.4.2 --driver-memory 512m --driver-cores 1 --master local[1]

# читаю csv файл
raw_files = spark \
    .read \
    .format("csv") \
    .schema(schema) \
    .options(path="csv_stream", header=True) \
    .load()
# убираю пробелы в названиях колонок чтобы загрузить в кассандру
cass_patients = raw_files.select("id", "radius_mean", "texture_mean",
                                 "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                                 "concavity_mean", F.col("concave points_mean").alias("concave_points_mean"),
                                 "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
                                 "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                                 F.col("concave points_se").alias("concave_points_se"), "symmetry_se",
                                 "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                                 "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                 F.col("concave points_worst").alias("concave_points_worst"), "symmetry_worst",
                                 "fractal_dimension_worst")
# записываю данные в кассандру
cass_patients.write \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="patients", keyspace="task8_course") \
    .mode("append") \
    .save()
# для проверки записи читаю данные из своей таблицы
patients_df = spark.read \
    .format("org.apache.spark.sql.cassandra") \
    .options(table="patients", keyspace="task8_course") \
    .load()
# меняю названия колонок обратно, чтобы они совпадали со схемой
cass_patients_df = patients_df.select("id", "radius_mean", "texture_mean",
                                      "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean",
                                      "concavity_mean", F.col("concave_points_mean").alias("concave points_mean"),
                                      "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se",
                                      "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se",
                                      F.col("concave_points_se").alias("concave points_se"), "symmetry_se",
                                      "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst",
                                      "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst",
                                      F.col("concave_points_worst").alias("concave points_worst"), "symmetry_worst",
                                      "fractal_dimension_worst")

pipeline_model = PipelineModel.load("my_LR_model_patients")


def writer_logic(df, epoch_id):
    df.persist()
    print("---------I've got new batch--------")
    print("This is what I've got from Kafka:")
    df.show()
    predict = pipeline_model.transform(df)
    predict_short = predict.select("id", F.col("category").alias("diagnosis"))
    patients_list_df = df.select("id").distinct()
    patients_list_rows = patients_list_df.collect()
    patients_list = map(lambda x: str(x.__getattr__("id")), patients_list_rows)
    where_string = " id = " + " or id = ".join(patients_list)
    print("These ids from Cassandra have dublicates in kafka:")
    dublicate_cass = cass_patients_df.where(where_string)
    dublicate = dublicate_cass.select("id")
    dublicate.show()
    print("Here is what I've got after model transformation:")
    predict_short.show()
    df.unpersist()


# связываем источник Кафки и foreachBatch функцию
stream = patients_flat \
    .writeStream \
    .trigger(processingTime='100 seconds') \
    .foreachBatch(writer_logic) \
    .option("checkpointLocation", "checkpoints/patients_unknown_checkpoint")

# поехали
s = stream.start()
s.stop()

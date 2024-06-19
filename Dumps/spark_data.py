import json
import os
from concurrent.futures import ProcessPoolExecutor

import findspark

findspark.init()
findspark.find()
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, IntegerType, LongType
#from pyspark.sql.window import Window as W

class SparkData:
    PATH_DUMP: str = './Dumps/KGDump/'
    PATH_EMBEDDINGS: str = 'D:/Embeddings/' #'D:/df_embs_avg_sent_trans/' #
    PATH_AUX: str = './Embeddings/aux/'
    PATH_EMBEDDINGS_LITERALS = PATH_EMBEDDINGS + "emb_lits_"
    PATH_EMBEDDINGS_ENTITIES_TEMP = PATH_EMBEDDINGS + "emb_ent_temp_"
    PATH_EMBEDDINGS_ENTITIES = PATH_EMBEDDINGS + "emb_ent_"
    PATH_TREES = PATH_EMBEDDINGS + "trees"

    def __init__(self):

        self.sparkSession = (SparkSession.builder.master('local[16]')
                             .config("spark.driver.memory", "20g")
                             .config('spark.executor.memory', '5G')
                             .config('spark.driver.maxResultSize', '3G')
                             .config('spark.driver.host', 'localhost')
                             .config("spark.local.dir", "d:/tmp")
                             .appName('dbpedia').getOrCreate())
        self.load_data_frames_from_dump()

        self.schema_df_embeddings = StructType([
            StructField('s', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True)
        ])

        self.schema_df_embeddings_full = StructType([
            StructField('s', StringType(), True),
            StructField('p', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True),
        ])

        self.schema_df_embeddings_full_top_n = StructType([
            StructField('s', StringType(), True),
            StructField('p', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True),
            StructField('count', LongType(), True),
            StructField('top_n', IntegerType(), True)
        ])

    def load_data_frames_from_dump(self):
        path_all = SparkData.PATH_DUMP + "all_entities"
        if not os.path.exists(path_all):

            files = [SparkData.PATH_DUMP + f for f in os.listdir(SparkData.PATH_DUMP) if f.endswith('.csv.bz2')]
            df = self.sparkSession.read.options(delimiter="|", header=True).csv(files[0])
            for file in files[1:]:
               df = df.union(self.sparkSession.read.options(delimiter="|", header=True).csv(file))
            #self.df_main = df

            df_main = (df.select('s', 'p', 'o')
                  .distinct().withColumn('id', monotonically_increasing_id())
                  .select('id', 's', 'p', 'o'))
            df_main.write.options(header=True, delimiter='|', compression="gzip").csv(path_all)
        self.df_main = self.sparkSession.read.options(delimiter="|", header=True).csv(path_all)

        path_e_l = SparkData.PATH_DUMP + "entities_literals"
        if not os.path.exists(path_e_l):
            df = (self.df_main.filter(col('o').rlike("(?i)^*@en .$"))
                  #.select('id', 's', 'p', 'o')
                  #.distinct().withColumn('id', monotonically_increasing_id())
                  #.select('id', 's', 'p', 'o')
                  )

            df.write.options(header=True, delimiter='|', compression="gzip").csv(path_e_l)
        self.df_literals = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_l)

        path_e_e = SparkData.PATH_DUMP + "entities_entities"
        if not os.path.exists(path_e_e):
            df = (self.df_main.filter(col('o').rlike("^http")).sort('s'))

            df.write.options(header=True, delimiter='|', compression="gzip").csv(path_e_e)
        self.df_entities = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_e)

        path_disambiguation = SparkData.PATH_DUMP + "entities_disambiguation"
        if not os.path.exists(path_disambiguation):
            files = [SparkData.PATH_DUMP + f for f in os.listdir(SparkData.PATH_DUMP) if f.endswith('.csv.bz2') and 'disambiguation' in f]

            if len(files) > 0:
                df = self.sparkSession.read.options(delimiter="|", header=True).csv(files[0])
                for file in files[1:]:
                    df = df.union(self.sparkSession.read.options(delimiter="|", header=True).csv(file))
                df.write.options(header=True, delimiter='|', compression="gzip").csv(path_disambiguation)

            # if len(files) > 0:
            #     df_disambiguation = self.sparkSession.read.options(delimiter="|", header=True).csv(files[0])

        if os.path.exists(path_disambiguation):
            self.df_disambiguation = self.sparkSession.read.options(delimiter="|", header=True).csv(path_disambiguation)

    def load_Embeddings(self):
        df_emb_entities = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_LITERALS)
        df_emb_literals = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_ENTITIES_TEMP)
        df = df_emb_literals.union(df_emb_entities)
        return df

    def load_Embeddings_path(self, path):
        return self.sparkSession.read.schema(self.schema_df_embeddings_full).json(path)

import json
import os
from concurrent.futures import ProcessPoolExecutor

import findspark

findspark.init()
findspark.find()
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, IntegerType
#from pyspark.sql.window import Window as W
from pyspark.sql import functions as F




class Data:
    PATH_DUMP: str = './Dump/'
    PATH_EMBEDDINGS: str = './Embeddings/'
    PATH_EMBEDDINGS_LITERALS = PATH_EMBEDDINGS + "embeddings_literals_full"
    PATH_EMBEDDINGS_ENTITIES_TEMP = PATH_EMBEDDINGS + "embeddings_entities_temp"
    PATH_EMBEDDINGS_ENTITIES = PATH_EMBEDDINGS + "embeddings_entities_full"
    PATH_TREES = PATH_EMBEDDINGS + "trees"

    def __init__(self, kg, check_on_endpoint=False):

        self.sparkSession = (SparkSession.builder.master('local[*]')
                             .config("spark.driver.memory", "20g")
                             .config('spark.executor.memory', '5G')
                             .config('spark.driver.host', 'localhost')
                             .appName('dbpedia').getOrCreate())
        self.check_on_endpoint = check_on_endpoint
        self.kg = kg
        self.load_data_frames_from_dump()

        self.schema_df_embeddings = StructType([
            StructField('s', StringType(), True),
            StructField('o', StringType(), True),
            StructField('typeObject', StringType(), True),
            StructField("embedding", ArrayType(FloatType(), True), True)
        ])

    def load_data_frames_from_dump(self):
        path_all = Data.PATH_DUMP + "all_entities"
        if not os.path.exists(path_all):

            files = [Data.PATH_DUMP + f for f in os.listdir(Data.PATH_DUMP) if f.endswith('.csv')]
            #self.df = self.sparkSession.read.options(delimiter="|", header=True).csv(files[2])
            df = self.sparkSession.read.options(delimiter="|", header=True).csv(files[0])
            for file in files[1:]:
               df = df.union(self.sparkSession.read.options(delimiter="|", header=True).csv(file))
            self.df_main = df

            df_main = (df.select('s', 'p', 'o')
                  .distinct().withColumn('id', monotonically_increasing_id())
                  .select('id', 's', 'p', 'o'))
            df_main.write.options(header=True, delimiter='|').csv(path_all)
        self.df_main = self.sparkSession.read.options(delimiter="|", header=True).csv(path_all)


        path_e_l = Data.PATH_DUMP + "entities_literals"
        if not os.path.exists(path_e_l):
            df = (self.df_main.filter(F.col('o').rlike("(?i)^*@en .$"))
                  #.select('id', 's', 'p', 'o')
                  #.distinct().withColumn('id', monotonically_increasing_id())
                  #.select('id', 's', 'p', 'o')
                  )

            df.write.options(header=True, delimiter='|').csv(path_e_l)
        self.df_literals = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_l)

        path_e_e = Data.PATH_DUMP + "entities_entities"
        if not os.path.exists(path_e_e):
            df = (self.df_main.filter(F.col('o').rlike("^http"))
                  #.select('s', 'p', 'o')
                  .sort('s')
                  #.distinct().withColumn('id', monotonically_increasing_id())
                  #.select('id', 's', 'p', 'o')
                  )

            df.write.options(header=True, delimiter='|').csv(path_e_e)
        self.df_entities = self.sparkSession.read.options(delimiter="|", header=True).csv(path_e_e)

    def load_Embeddings(self):
        df_emb_entities = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_LITERALS)
        df_emb_literals = self.sparkSession.read.schema(self.schema_df_embeddings).json(self.PATH_EMBEDDINGS_ENTITIES)
        df = df_emb_literals.union(df_emb_entities)#.withColumn('id', monotonically_increasing_id())

        #df.createOrReplaceTempView('df1')
        #self.sparkSession.sql('select  row_number() over(order by id) as num, * from df1').show()


        # data_len = df.count()
        # labels_ids = F.udf(
        #     lambda indx: ids[indx - 1],
        #     IntegerType())
        # ids = list(range(1, data_len+1, 1))
        # df = df.withColumn("num_id", row_number().over(Window.orderBy(monotonically_increasing_id())))
        # df = df.withColumn('id', labels_ids('num_id')).drop('num_id')


        # schema = StructType([
        #     StructField('id_l', StringType(), True),
        #     StructField('centroid', StringType(), True),
        # ])
        # df = df.join(self.sparkSession.createDataFrame([(i + 1, ids[i]) for i in range(len(ids))], schema), F.col('id') == F.col('id_l'))

        # w = W().orderBy(F.lit('A'))
        # w = W().orderBy(F.monotonically_increasing_id())
        # df = df.withColumn('id', F.row_number().over(w))

        #df = df.repartition(4, "id")



        #b.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

        #windowSpec = W.orderBy("id")
        #df = df.withColumn("id", F.row_number().over(windowSpec)).show()

        return df

        #return df_emb_literals.union(df_emb_entities).withColumn('id', monotonically_increasing_id())









    # def get_triples(self, sentence):
    #     sentence = re.sub('\r?\n', ' ', sentence)
    #     sentence = re.findall('\{.*?\}', sentence)[0][1:-1].rstrip(". ").strip()  # Get substring inside { and }
    #     sentence = re.sub('\r? \. ', '|||', sentence)  # Replace space_dot_space for |||
    #     sentence = re.sub('\r?\. <', '|||<', sentence)  # Replace dot_< for |||
    #     sentence = re.sub('\r?\. \?', '|||?', sentence)  # Replace dot_? for |||
    #     sentence = re.sub('\r?>\. ', '|||?', sentence)  # Replace dot_? for |||
    #     sentence = re.sub('\r?>|<', '', sentence)  # Remove < OR >
    #     sentence = sentence.split('|||')
    #     return sentence

    # def get_dump(self):
    #     path_files = "./Parser/dbpedia_files.txt"
    #     path_dir = "./Temp/"
    #
    #     lines = pathlib.Path(path_files).read_text().splitlines()
    #     Parallel(n_jobs=6)(
    #         delayed(Data.get_file)(remote_file, path_dir) for remote_file in lines if '#' not in remote_file)
    #
    # def get_file(remote_file, path_dir):
    #     if '#' in remote_file:
    #         return
    #     print("Downloading file " + remote_file)
    #     local_file = path_dir + remote_file.split('/')[-1]
    #     if os.path.isfile(local_file):
    #         print('File exists')
    #     else:
    #         data = requests.get(remote_file)
    #         with open(local_file, 'wb') as file:
    #             try:
    #                 file.write(data.content)
    #             except:
    #                 print("fail")

    # def load_lcquad_dataset(self):
    #     path_dataset = './Datasets/LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json'
    #
    #     with open(path_dataset, encoding='utf8') as data_file:
    #         dataset = json.load(data_file)
    #
    #     return dataset

    # def load_conll_dataset(self):
    #     path_dataset = './Datasets/Conll/aida-yago2-dataset/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv'
    #
    #     with open(path_dataset, encoding='utf8') as data_file:
    #         dataset = data_file.readlines()
    #
    #     return dataset

    def load_lcquad_grounth_truth(self):
        entity_mapping = {}
        predicate_mapping = {}
        path_dataset = './Datasets/LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json'

        with open(path_dataset, encoding='utf8') as data_file:
            dataset = json.load(data_file)

        for text in dataset:
            for mapping in text['entity mapping']:
                entity_mapping[mapping['label']] = mapping['uri']

        for text in dataset:
            for mapping in text['predicate mapping']:
                if 'label' in mapping:
                    predicate_mapping[mapping['label']] = mapping['uri']

        return entity_mapping, predicate_mapping

    def load_conll_grounth_truth(self):
        entity_mapping = {}
        predicate_mapping = {}

        path_dataset = './Datasets/Conll/aida-yago2-dataset/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv'

        with open(path_dataset, encoding='utf8') as data_file:
            dataset = data_file.readlines()

        for line in dataset:
            if "http://en.wikipedia.org/wiki/" in line:
                splitted_line = line.split("\t")
                try:
                    entity_mapping[splitted_line[2]] = splitted_line[4]
                except:
                    print(splitted_line)

        return entity_mapping, predicate_mapping

    # def load_entities(self, from_url_dump=True):
    #     directory = './Temp/'
    #     all_entities_file = directory + "all_entities.pkl"
    #     entities = []
    #     relations = []
    #
    #     if not os.path.isfile(all_entities_file):
    #
    #         if from_url_dump:
    #
    #             files = [f for f in os.listdir(directory) if f.endswith('.bz2')]
    #
    #             for i in range(len(files)):
    #                 source_file = bz2.BZ2File('./Temp/' + files[i], "r")
    #
    #                 source_file.readline()  # It ignores the first line
    #                 next_line = source_file.readline()
    #                 while True:
    #                     line = next_line
    #                     next_line = source_file.readline()
    #                     if len(next_line) == 0:  # This check if the current line is the last and ignore
    #                         break
    #                     txt = line.decode("utf-8")
    #                     entities.append(txt.split(' ')[0][1:-1])
    #
    #         else:
    #
    #             all_triples = [self.get_triples(text['sparql_query']) for text in self.lcquad_dataset]
    #             for triples in all_triples:
    #                 # print('--')
    #                 for triple in triples:
    #                     # print(triple)
    #                     if not 'filter' in triple.lower():
    #                         rdf_triple = triple.split(' ')
    #
    #                     if 'http://' in rdf_triple[0]:
    #                         if 'resource' in rdf_triple[0]:
    #                             entities.append(rdf_triple[0])
    #                         else:
    #                             relations.append(rdf_triple[0])
    #
    #                     if 'http://' in rdf_triple[2]:
    #                         if 'resource' in rdf_triple[2]:
    #                             entities.append(rdf_triple[2])
    #                         else:
    #                             relations.append(rdf_triple[2])
    #
    #                     if 'http://' in rdf_triple[1]:
    #                         relations.append(rdf_triple[1])
    #
    #         entities = list(set(entities))
    #         relations = list(set(relations))
    #
    #         all_entities_relations = sorted(list(set(entities + relations)))
    #
    #         if self.check_on_endpoint:
    #             print('Checking entities in Virtuoso ...')
    #             try:
    #                 print('Opening entities_not_in_virtuoso.txt.')
    #                 with open("entities_not_in_virtuoso.txt", 'r', encoding='utf8') as file:
    #                     print('Reading lines from entities_not_in_virtuoso.txt.')
    #                     entities_not_in_list = file.readlines()
    #                     index_entities_not_in_list = [all_entities_relations.index(e[:-1]) for e in
    #                                                   entities_not_in_list]
    #             except Exception as e:
    #                 print(e)
    #                 print('File entities_not_in_virtuoso.txt does not exists.')
    #                 entities_not_in_list, index_entities_not_in_list = self.is_exist(all_entities_relations)
    #                 print('Creating file entities_not_in_virtuoso.txt.')
    #                 try:
    #                     with open("entities_not_in_virtuoso.txt", 'w', encoding='utf8')as file:
    #                         for element in entities_not_in_list:
    #                             file.write(element + '\n')
    #                 except Exception as ex:
    #                     print(ex)
    #                     logging.info(f"Message: {ex}")
    #                 logging.info(f"Message: {e}")
    #
    #             for i in sorted(index_entities_not_in_list, reverse=True):
    #                 del all_entities_relations[i]
    #         file = open(all_entities_file, 'wb')
    #         pickle.dump(all_entities_relations, file)
    #         file.close()
    #     else:
    #         file = open(all_entities_file, 'rb')
    #         all_entities_relations = pickle.load(file)
    #         file.close()
    #
    #     return all_entities_relations, entities, relations

    def is_exist(self, entities):
        queries = [
            f"ASK WHERE {{ <{entity}> ?p ?o . }}" for entity in entities
        ]

        responses = []
        with ProcessPoolExecutor(max_workers=10) as executor:
            for r in executor.map(self.kg.get_from_kg, queries):
                responses.append(r)
        responses = [res["boolean"] for res in responses]

        index_entities_not_in_list = [i for i, val in enumerate(responses) if not val]
        return [entities[i] for i in index_entities_not_in_list], index_entities_not_in_list



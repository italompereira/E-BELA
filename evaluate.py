import bz2
import json
import logging
import math
import os
import pickle
import re
import time
import requests
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
from random import sample
from Encoders.encoder import Encoder
from data import Data
from kg import KG
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.window import Window as W
from pyspark.sql import functions as F
from pyspark.sql.functions import *
import random
import numpy as np
from sklearn.neighbors import KDTree
import builtins


class Evaluate:

    ONLY_URIS = 0
    ONLY_LITERALS = 1
    ALL = 2

    def __init__(self, path_endpoint, data, embeddings_of):
        #self.embeddings_list = self.get_embeddings(embeddings_file, embeddings_of)
        self.embeddings_of = embeddings_of
        #if embeddings_of != Evaluate.ONLY_URIS:
        #    self.literal_relations = self.get_literals_relations(literal_relations_file)

        try:
            self.kg = KG(path_endpoint, is_remote=True, skip_verify=True)
        except Exception as ex:
            self.kg = None
            print(ex)
            logging.info(f"Message: {ex}")

        #self.kg = KG(path_endpoint, is_remote=True, skip_verify=True)
        self.data = data
        #self.entities, _, _ = data.load_entities()
        self.entities_mapping, self.predicates_mapping = data.load_lcquad_grounth_truth()
        # self.entities_mapping, self.predicates_mapping = data.load_conll_grounth_truth()
        #self.path_files = path_files

    def __call__(self, *args, **kwargs):
        print('Evaluating embeddings.')
        self.evaluate()

    def evaluate(self, ):
        transformer = Encoder(self.kg, self.data)
        print("Embedding " + str(len(self.entities_mapping)) + " entities from Ground Truth.")
        vectors = transformer.get_embedding(list(self.entities_mapping))

        #self.generate_blocking()


        top_k = self.generate_similarity_matrix(vectors)

        y_true = [i for i in self.entities_mapping.values()]

        y_pred = []
        for i in range(len(vectors)):
            if i % 10 == 0:
                print(str(i) + " vectors processed of " + str(len(vectors)), end=".\t")
            y_pred.append(self.get_similars_to(i, top_k, 10))
            if i % 100 == 0:
                print("\n")

        print('Accuracy: ')
        print(sum(1 for x, y in zip(y_true, y_pred) if x == y) / len(y_true))

    def generate_blocking(self):
        print("Blocking")

        if not os.path.exists(self.data.PATH_TREES):

            # rng = np.random.default_rng()
            # n_dim = 512
            n_clusters = 250
            # sample = rng.uniform([0] * n_dim, [1] * n_dim, size=(n_clusters, n_dim)).tolist()
            # centroids = tf.stack(sample)

            df_embeddings = self.data.load_Embeddings()#.limit(500)
            data_len = df_embeddings.count()
              # math.ceil(math.sqrt(float(data_len)))

            # Add a column id to dataframe of embeddings
            w = W().orderBy(F.lit('A'))
            w = W().orderBy(F.monotonically_increasing_id())
            df_embeddings_aux = df_embeddings.withColumn('id', F.row_number().over(w))

            # Initialize the centroids
            #rand_index = random.sample(range(1,data_len+1), n_clusters)
            #centroids = [tf.constant(row[3]) for row in df_embeddings.filter(col('id').isin(rand_index)).toLocalIterator()]


            fraction_size = 1/(data_len/(n_clusters+20))
            centroids = [tf.constant(row[3]) for row in df_embeddings.sample(fraction=fraction_size, seed=3).limit(n_clusters).toLocalIterator()]
            # centroids = [tf.constant(row[3]) for row in df_embeddings.sampleBy("typeObject", fractions={'E': fraction_size, 'L': fraction_size}, seed=10).sort('s').limit(n_clusters).toLocalIterator()]
            number_rows_per_centroid = math.ceil(data_len / n_clusters)

            #schema_df_embeddings = self.data.schema_df_embeddings
            #schema_df_embeddings.add('id', IntegerType(), True)

            changed = True
            iteration = 0
            while changed and iteration < 10:
                print("Iteration " + str(iteration))
                df_iterator = df_embeddings.toLocalIterator()
                iteration += 1
                changed = False

                s = [-1] * data_len

                # appends elements to centroids
                full_centroids = [0] * n_clusters
                for i in range(data_len):
                    if i % 100000 == 0:
                        print(str(i) + "\t vectors processed of " + str(data_len), end=".\n")

                    row = next(iter(df_iterator))
                    distance = self.calc_distance(tf.constant(row.asDict()['embedding']), centroids, axis=1)
                    # for centroid in centroids:
                    #     distance.append(self.calc_distance(tf.constant(row.asDict()['embedding']), centroid))

                    # centroid_index = tf.argmin(distance)
                    # centroids_data[centroid_index].append(i)

                    centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_clusters)[1].numpy()

                    for centroid_index in centroid_indexes:
                        if full_centroids[centroid_index] < number_rows_per_centroid: # or s.count(centroid_index) <= number_rows_per_centroid:
                            s[i] = centroid_index
                            full_centroids[centroid_index] += 1
                            # if s.count(centroid_index) >= number_rows_per_centroid:
                            #     full_centroids.append(centroid_index)
                            break
                        #else:
                            #print(f'fail - i: {i}, index: {centroid_index}')

                    if s[i] == -1:
                        print('no one index ')


                # Adds the centroids ids to embeddings dataframe
                schema = StructType([
                    StructField('id', StringType(), True),
                    StructField('centroid', StringType(), True),
                ])
                df_s = self.data.sparkSession.createDataFrame([(i + 1, s[i].item()) for i in range(len(s))], schema)
                df_embeddings_block = df_embeddings_aux.join(df_s, df_embeddings_aux.id == df_s.id).drop(df_s.id)

                #df_embeddings_block.write.json(self.data.PATH_EMBEDDINGS + "embeddings_with_blocks")

            os.mkdir(self.data.PATH_TREES)
            with open(self.data.PATH_TREES + '/centroids.pickle', 'wb') as handle:
                pickle.dump(centroids, handle, protocol=pickle.HIGHEST_PROTOCOL)

            for centroid in range(len(centroids)):
                print(f'Saving trees: {centroid}')
                centroid_data = df_embeddings_block.filter(col('centroid') == centroid).collect()

                centroid_data_embeddings = [row.asDict()['embedding'] for row in centroid_data]
                centroid_data_ids = [row.asDict()['id'] for row in centroid_data]
                centroid_data_objects = [row.asDict()['o'] for row in centroid_data]

                tree = KDTree(centroid_data_embeddings, leaf_size=2)

                with open(self.data.PATH_TREES + '/' + str(centroid) + '_tree' + '.pickle', 'wb') as handle:
                    pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(self.data.PATH_TREES + '/' + str(centroid) + '_indexes' + '.pickle', 'wb') as handle:
                    pickle.dump(centroid_data_ids, handle, protocol=pickle.HIGHEST_PROTOCOL)

                with open(self.data.PATH_TREES + '/' + str(centroid) + '_objects' + '.pickle', 'wb') as handle:
                    pickle.dump(centroid_data_objects, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.data.PATH_TREES + '/centroids.pickle', 'rb') as handle:
            centroids = pickle.load(handle)

        return centroids

                # # adjusts the distribuition of the centroids
            # centroids_distribution = df_embeddings_block.groupBy('centroid').count().withColumn("centroid",col('centroid').cast('integer')).sort(col('centroid'), ascending=False).collect()
            # centroids_to_remove = []
            # centroids_to_redistribute = []
            # for row in centroids_distribution:
            #     if row.asDict()['count'] <= number_rows_per_centroid - (number_rows_per_centroid*0.1):
            #         centroids_to_remove.append(row.asDict()['centroid'])
            #     elif row.asDict()['count'] >= number_rows_per_centroid + (number_rows_per_centroid*0.1):
            #         centroids_to_remove.append(row.asDict()['centroid'])
            #         centroids_to_redistribute.append(row.asDict()['centroid'])
            #
            # for index in centroids_to_remove:
            #     del centroids[index]
            #
            # for i in range(len(centroids_to_redistribute)):
            #     print(f'Redistribute Centroids {i} : {len(centroids_to_redistribute)}')
            #     index = centroids_to_redistribute[i]
            #     len_centroid_index = df_embeddings_block.filter(col('centroid') == index).count()
            #     sample_size = math.ceil(len_centroid_index / number_rows_per_centroid)
            #     fraction_size = 1 / (len_centroid_index / (sample_size))
            #     new_centroids = [tf.constant(row[3]) for row in df_embeddings_block.filter(col('centroid') == index).sampleBy("typeObject", fractions={'E': fraction_size, 'L': fraction_size}, seed=10).sort('s').limit(sample_size).toLocalIterator()]
            #     if len(new_centroids) > 0:
            #         changed = True
            #         centroids = centroids + new_centroids


            # # adjusts the center of the centroids
            # for i in range(n_clusters):
            #     vectors = [row.asDict()['embedding'] for row in df_embeddings_block.filter(col('centroid') == i).select('embedding').toLocalIterator()]
            #     mean = tf.math.reduce_mean(tf.stack(vectors, axis=1), 1)
            #     if not tf.math.reduce_all(tf.equal(centroids[i], mean)).numpy():
            #         centroids[i] = mean
            #         changed = True

        # centroids_dict = []
        # for i in range(n_clusters):
        #     centroids_dict.append((centroids[i], centroids_data[i]))
        #
        # #     file = open(file, 'wb')
        # #     pickle.dump(centroids_dict, file)
        # #     file.close()
        # #     print("Finishing")
        # #
        # # else:
        # #     file = open(file, 'rb')
        # #     centroids_dict = pickle.load(file)
        # #     file.close()
        # return centroids_dict

    # def generate_blocking_data_centers(self):
    #     # file = self.path_files + "blocking.pkl"
    #     print("Blocking")
    #     # if not os.path.isfile(file):
    #     embeddings_list = self.data.load_Embeddings()#.limit(5000)
    #
    #
    #
    #
    #
    #     #data = [tf.constant(i.asDict()['embedding']) for i in embeddings_list.toLocalIterator()]
    #     data_len = embeddings_list.count()
    #     n_clusters = 250  # math.ceil(math.sqrt(float(data_len)))
    #
    #     # Initialize the centroids
    #     centroids = [tf.constant(row[3]) for row in embeddings_list.sample(fraction=1.0).limit(n_clusters).toLocalIterator()]
    #
    #     # Initialize dataframes to store rows of centroids
    #     schema_df_embeddings = self.data.schema_df_embeddings
    #     schema_df_embeddings.add('id', IntegerType(), True)
    #
    #     n_shared = 10
    #     centroids_data_aux_size = 5000
    #
    #
    #
    #
    #
    #
    #     changed = True
    #     iteration = 0
    #     while changed and iteration < 10:
    #         print("Iteration " + str(iteration))
    #         df_iterator = embeddings_list.toLocalIterator()
    #         iteration += 1
    #         changed = False
    #         centroids_data = []
    #         centroids_data_aux = [[] for i in range(n_clusters)]
    #         for i in range(n_clusters):
    #             centroids_data.append(self.data.sparkSession.createDataFrame([], schema_df_embeddings))
    #         # appends elements to centroids
    #         for i in range(data_len):
    #             if i % 10 == 0:
    #                 print(str(i) + "\t vectors processed of " + str(data_len), end=".\n")
    #                 if (i > 0) & (i % centroids_data_aux_size == 0):
    #                     for i in range(len(centroids_data_aux)):
    #                         centroids_data[i] = centroids_data[i].union(
    #                             self.data.sparkSession.createDataFrame(centroids_data_aux[i], schema_df_embeddings))
    #                     centroids_data_aux = [[] for i in range(n_clusters)]
    #
    #             row = next(iter(df_iterator))
    #             distance = []
    #             for centroid in centroids:
    #                 distance.append(self.calc_distance(tf.constant(row.asDict()['embedding']), centroid))
    #
    #             # centroid_index = tf.argmin(distance)
    #             # centroids_data[centroid_index].append(i)
    #
    #             centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_shared)[1].numpy()
    #             for centroid_index in centroid_indexes:
    #                 centroids_data_aux[centroid_index].append(tuple(row.asDict().values()))
    #
    #
    #
    #         # adjusts the center of the centroids
    #         for i in range(n_clusters):
    #             vectors = [row.asDict()['embedding'] for row in centroids_data[i].select('embedding').toLocalIterator()]
    #             mean = tf.math.reduce_mean(tf.stack(vectors, axis=1), 1)
    #             if not tf.math.reduce_all(tf.equal(centroids[i], mean)).numpy():
    #                 centroids[i] = mean
    #                 changed = True
    #
    #     centroids_dict = []
    #     for i in range(n_clusters):
    #         centroids_dict.append((centroids[i], centroids_data[i]))
    #
    #     #     file = open(file, 'wb')
    #     #     pickle.dump(centroids_dict, file)
    #     #     file.close()
    #     #     print("Finishing")
    #     #
    #     # else:
    #     #     file = open(file, 'rb')
    #     #     centroids_dict = pickle.load(file)
    #     #     file.close()
    #     return centroids_dict
    #
    #
    # def generate_blocking_old(self):
    #     # file = self.path_files + "blocking.pkl"
    #     print("Blocking")
    #     # if not os.path.isfile(file):
    #     embeddings_list = self.data.load_Embeddings().limit(500).toLocalIterator()
    #     #embeddings_list = [row[0] for row in self.data.load_Embeddings().limit(500).select('embedding').toLocalIterator()]
    #     data = [tf.constant(i.asDict()['embedding']) for i in embeddings_list]
    #     data_len = len(data)
    #     n_clusters = 50  # math.ceil(math.sqrt(float(data_len)))
    #     n_shared = 10
    #     centroids = sample(data, n_clusters)
    #
    #     changed = True
    #     iteration = 0
    #     while changed and iteration < 10:
    #         print("Iteration " + str(iteration))
    #         iteration += 1
    #         changed = False
    #         centroids_data = [[] for i in range(n_clusters)]
    #         # appends elements to centroids
    #         for i in range(data_len):
    #             if i % 1000 == 0:
    #                 print(str(i) + "\t vectors processed of " + str(data_len), end=".\n")
    #
    #             vector = data[i]
    #             distance = []
    #             for centroid in centroids:
    #                 distance.append(self.calc_distance(vector, centroid))
    #
    #             # centroid_index = tf.argmin(distance)
    #             # centroids_data[centroid_index].append(i)
    #
    #             centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_shared)[1].numpy()
    #             for centroid_index in centroid_indexes:
    #                 centroids_data[centroid_index].append(i)
    #
    #         # adjusts the center of the centroids
    #         for i in range(n_clusters):
    #             vectors = [data[j] for j in centroids_data[i]]
    #             mean = tf.math.reduce_mean(tf.stack(vectors, axis=1), 1)
    #             if not tf.math.reduce_all(tf.equal(centroids[i], mean)).numpy():
    #                 centroids[i] = mean
    #                 changed = True
    #
    #     centroids_dict = []
    #     for i in range(n_clusters):
    #         centroids_dict.append((centroids[i], centroids_data[i]))
    #
    #     #     file = open(file, 'wb')
    #     #     pickle.dump(centroids_dict, file)
    #     #     file.close()
    #     #     print("Finishing")
    #     #
    #     # else:
    #     #     file = open(file, 'rb')
    #     #     centroids_dict = pickle.load(file)
    #     #     file.close()
    #     return centroids_dict


        # file = self.path_files + "blocking.pkl"
        # print("Blocking")
        # if not os.path.isfile(file):
        #     embeddings_list = self.embeddings_list
        #     data = [i[1] for i in embeddings_list]
        #     data_len = len(data)
        #     n_clusters = 50  # math.ceil(math.sqrt(float(data_len)))
        #     n_shared = 10
        #     centroids = sample(data, n_clusters)
        #
        #     changed = True
        #     iteration = 0
        #     while changed and iteration < 10:
        #         print("Iteration " + str(iteration))
        #         iteration += 1
        #         changed = False
        #         centroids_data = [[] for i in range(n_clusters)]
        #         # appends elements to centroids
        #         for i in range(data_len):
        #             if i % 1000 == 0:
        #                 print(str(i) + "\t vectors processed of " + str(data_len), end=".\n")
        #
        #             vector = data[i]
        #             distance = []
        #             for centroid in centroids:
        #                 distance.append(self.calc_distance(vector, centroid))
        #
        #             # centroid_index = tf.argmin(distance)
        #             # centroids_data[centroid_index].append(i)
        #
        #             centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_shared)[1].numpy()
        #             for centroid_index in centroid_indexes:
        #                 centroids_data[centroid_index].append(i)
        #
        #         # adjusts the center of the centroids
        #         for i in range(n_clusters):
        #             vectors = [data[j] for j in centroids_data[i]]
        #             mean = tf.math.reduce_mean(tf.stack(vectors, axis=1), 1)
        #             if not tf.math.reduce_all(tf.equal(centroids[i], mean)).numpy():
        #                 centroids[i] = mean
        #                 changed = True
        #
        #     centroids_dict = []
        #     for i in range(n_clusters):
        #         centroids_dict.append((centroids[i], centroids_data[i]))
        #
        #     file = open(file, 'wb')
        #     pickle.dump(centroids_dict, file)
        #     file.close()
        #     print("Finishing")
        #
        # else:
        #     file = open(file, 'rb')
        #     centroids_dict = pickle.load(file)
        #     file.close()
        # return centroids_dict

    def load_tree(self, centroid):
        with open(self.data.PATH_TREES + '/' + str(centroid) + '_tree' + '.pickle', 'rb') as handle:
            tree = pickle.load(handle)

        with open(self.data.PATH_TREES + '/' + str(centroid) + '_indexes' + '.pickle', 'rb') as handle:
            centroid_data_ids = pickle.load(handle)

        with open(self.data.PATH_TREES + '/' + str(centroid) + '_objects' + '.pickle', 'rb') as handle:
            objects = pickle.load(handle)

        return tree, centroid_data_ids, objects
    def generate_similarity_matrix(self, vectors):
        print("Generating similarity matrix")

        start = time.time()
        centroids = self.generate_blocking()
        end = time.time()
        print(end - start)
        top_k = {}
        for i in range(len(vectors)):
            if i % 100 == 0:
                print(str(i) + " vectors processed of " + str(len(vectors)), end=".\n")
                print(end - start)
            start = time.time()
            distance_centroids = []
            for j in range(len(centroids)):
                distance_centroids.append(self.calc_distance(vectors[i], centroids[j]))

            index_centroid = tf.argmin(distance_centroids).numpy()

            tree, indexes, objects = self.load_tree(index_centroid)

            distance_array, indexes_distance = tree.query([vectors[i]], k=100)


            # embeddings_indexes = centroids_dict[index_centroid][1]
            #
            # distance_array = []
            # for index in embeddings_indexes:
            #     vector_a = vectors[i]
            #     vector_b = self.embeddings_list[index][1]
            #     distance_array.append(self.calc_distance(vector_a, vector_b))

            top_k_size = 100 if len(distance_array) >= 100 else len(distance_array[0])
            top_k[i] = [objects[index_distance] for index_distance in indexes_distance[0]][:top_k_size]
            end = time.time()
            #print(end - start)
        #     file = open(file, 'wb')
        #     pickle.dump(top_k, file)
        #     file.close()
        # else:
        #     file = open(file, 'rb')
        #     top_k = pickle.load(file)
        #     file.close()

        return top_k


        # file = self.path_files + "similarity_matrix.pkl"
        # print("Generating similarity matrix")
        # if not os.path.isfile(file):
        #     start = time.time()
        #     centroids_dict = self.generate_blocking()
        #     end = time.time()
        #     print(end - start)
        #     top_k = {}
        #     for i in range(len(vectors)):
        #         if i % 100 == 0:
        #             print(str(i) + " vectors processed of " + str(len(vectors)), end=".\n")
        #
        #         start = time.time()
        #         distance_centroids = []
        #         for j in range(len(centroids_dict)):
        #             distance_centroids.append(self.calc_distance(vectors[i], centroids_dict[j][0]))
        #
        #         index_centroid = tf.argmin(distance_centroids)
        #         embeddings_indexes = centroids_dict[index_centroid][1]
        #
        #         distance_array = []
        #         for index in embeddings_indexes:
        #             vector_a = vectors[i]
        #             vector_b = self.embeddings_list[index][1]
        #             distance_array.append(self.calc_distance(vector_a, vector_b))
        #
        #         top_k_size = 100 if len(distance_array) >= 100 else len(distance_array)
        #         top_k[i] = [embeddings_indexes[l] for l in
        #                     tf.math.top_k(tf.negative(distance_array), k=top_k_size)[1].numpy()]
        #         end = time.time()
        #         print(end - start)
        #     file = open(file, 'wb')
        #     pickle.dump(top_k, file)
        #     file.close()
        # else:
        #     file = open(file, 'rb')
        #     top_k = pickle.load(file)
        #     file.close()
        #
        # return top_k



    def get_similars_to(self, i, top_k, k):
        #embedded_entities = [i[0] for i in self.embeddings_list]
        #indexes = top_k[i][:k]
        ranking_list_aux = top_k[i] # [embedded_entities[i] for i in indexes]
        ranking_list = {}
        for i in range(len(ranking_list_aux)):
            res_entity = ranking_list_aux[i]
            # if  in not (res_entity.startswith("https:") or res_entity.startswith(
            #         "http:")):
            #     for entity in self.literal_relations[res_entity]:
            #         ranking_list[entity] = 1 / (i + 1) if entity not in ranking_list else ranking_list[entity] + 1 / (i + 1)
            # else:
            ranking_list[res_entity] = 1 / (i + 1) if res_entity not in ranking_list else ranking_list[res_entity] + 1 / (i + 1)

        return builtins.max(ranking_list, key=ranking_list.get)

    def calc_distance(self, vector_a, vector_b, axis=None):
        # Similaridade do cosseno
        # return tf.keras.losses.CosineSimilarity(axis=0)(vector_a, vector_b).numpy()

        # Distância Euclidiana
        return tf.norm(vector_a - vector_b, axis=axis, ord='euclidean')

    # def get_embeddings(self, embeddings_file, embeddings_of):
    #     embeddings_list = []
    #
    #     with open(embeddings_file, encoding='utf8') as embedding_file:
    #         lines = embedding_file.readlines()
    #
    #         for i in range(len(lines)):
    #
    #             if embeddings_of == Evaluate.ONLY_URIS:
    #                 if not (lines[i].startswith("https:") or lines[i].startswith("http:")):
    #                     continue
    #             elif embeddings_of == Evaluate.ONLY_LITERALS:
    #                 if (lines[i].startswith("https:") or lines[i].startswith("http:")):
    #                     continue
    #
    #             line_splitted = lines[i].split('|||')
    #             try:
    #                 embeddings_list.append(
    #                     (line_splitted[0], tf.constant([float(x) for x in line_splitted[1].split(' ')])))
    #             except Exception as e:
    #                 print(e)
    #                 logging.info(f"Message: {e}")
    #
    #     return embeddings_list
    #
    # def get_literals_relations(self, literals_relations_file):
    #     literals_relations = {}
    #
    #     with open(literals_relations_file, encoding='utf8') as literals_relations_file:
    #         lines = literals_relations_file.readlines()
    #
    #         for i in range(len(lines)):
    #
    #             line_splitted = lines[i].split('|||')
    #
    #             if line_splitted[0] in literals_relations:
    #                 for entity in line_splitted[1:]:
    #                     literals_relations[line_splitted[0]].append(entity.strip())
    #             else:
    #                 literals_relations[line_splitted[0]] = []
    #                 for entity in line_splitted[1:]:
    #                     literals_relations[line_splitted[0]].append(entity.strip())
    #
    #     return literals_relations
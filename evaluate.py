import bz2
import json
import logging
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


class Evaluate:

    ONLY_URIS = 0
    ONLY_LITERALS = 1
    ALL = 2

    def __init__(self, endpoint, embeddings_file, literal_relations_file, path_files, embeddings_of):
        self.embeddings_list = self.get_embeddings(embeddings_file, embeddings_of)
        if embeddings_of != Evaluate.ONLY_URIS:
            self.literal_relations = self.get_literals_relations(literal_relations_file)

        self.kg = KG(endpoint, is_remote=True, skip_verify=True)
        data = Data(self.kg)
        self.entities, _, _ = data.load_entities()
        self.entities_mapping, self.predicates_mapping = data.load_lcquad_grounth_truth()
        # self.entities_mapping, self.predicates_mapping = data.load_conll_grounth_truth()
        self.path_files = path_files

    def __call__(self, *args, **kwargs):
        print('Evaluating embeddings.')
        self.evaluate()

    def generate_blocking(self):
        file = self.path_files + "blocking.pkl"
        print("Blocking")
        if not os.path.isfile(file):
            embeddings_list = self.embeddings_list
            data = [i[1] for i in embeddings_list]
            data_len = len(data)
            n_clusters = 50  # math.ceil(math.sqrt(float(data_len)))
            n_shared = 10
            centroids = sample(data, n_clusters)

            changed = True
            iteration = 0
            while changed and iteration < 10:
                print("Iteration " + str(iteration))
                iteration += 1
                changed = False
                centroids_data = [[] for i in range(n_clusters)]
                # appends elements to centroids
                for i in range(data_len):
                    if i % 1000 == 0:
                        print(str(i) + "\t vectors processed of " + str(data_len), end=".\n")

                    vector = data[i]
                    distance = []
                    for centroid in centroids:
                        distance.append(self.calc_distance(vector, centroid))

                    # centroid_index = tf.argmin(distance)
                    # centroids_data[centroid_index].append(i)

                    centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_shared)[1].numpy()
                    for centroid_index in centroid_indexes:
                        centroids_data[centroid_index].append(i)

                # adjusts the center of the centroids
                for i in range(n_clusters):
                    vectors = [data[j] for j in centroids_data[i]]
                    mean = tf.math.reduce_mean(tf.stack(vectors, axis=1), 1)
                    if not tf.math.reduce_all(tf.equal(centroids[i], mean)).numpy():
                        centroids[i] = mean
                        changed = True

            centroids_dict = []
            for i in range(n_clusters):
                centroids_dict.append((centroids[i], centroids_data[i]))

            file = open(file, 'wb')
            pickle.dump(centroids_dict, file)
            file.close()
            print("Finishing")

        else:
            file = open(file, 'rb')
            centroids_dict = pickle.load(file)
            file.close()
        return centroids_dict

    def generate_similarity_matrix(self, vectors):
        file = self.path_files + "similarity_matrix.pkl"
        print("Generating similarity matrix")
        if not os.path.isfile(file):
            start = time.time()
            centroids_dict = self.generate_blocking()
            end = time.time()
            print(end - start)
            top_k = {}
            for i in range(len(vectors)):
                if i % 100 == 0:
                    print(str(i) + " vectors processed of " + str(len(vectors)), end=".\n")

                start = time.time()
                distance_centroids = []
                for j in range(len(centroids_dict)):
                    distance_centroids.append(self.calc_distance(vectors[i], centroids_dict[j][0]))

                index_centroid = tf.argmin(distance_centroids)
                embeddings_indexes = centroids_dict[index_centroid][1]

                distance_array = []
                for index in embeddings_indexes:
                    vector_a = vectors[i]
                    vector_b = self.embeddings_list[index][1]
                    distance_array.append(self.calc_distance(vector_a, vector_b))

                top_k_size = 100 if len(distance_array) >= 100 else len(distance_array)
                top_k[i] = [embeddings_indexes[l] for l in
                            tf.math.top_k(tf.negative(distance_array), k=top_k_size)[1].numpy()]
                end = time.time()
                print(end - start)
            file = open(file, 'wb')
            pickle.dump(top_k, file)
            file.close()
        else:
            file = open(file, 'rb')
            top_k = pickle.load(file)
            file.close()

        return top_k

    def evaluate(self, ):
        transformer = Encoder(self.kg)
        print("Embedding " + str(len(self.entities_mapping)) + " entities from Ground Truth.")
        vectors = transformer.fit_transform(list(self.entities_mapping))

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

    def get_similars_to(self, i, top_k, k):
        embedded_entities = [i[0] for i in self.embeddings_list]
        indexes = top_k[i][:k]
        ranking_list_aux = [embedded_entities[i] for i in indexes]
        ranking_list = {}
        for i in range(len(ranking_list_aux)):
            res_entity = ranking_list_aux[i]
            if not (res_entity.startswith("https:") or res_entity.startswith(
                    "http:")) and res_entity in self.literal_relations:
                for entity in self.literal_relations[res_entity]:
                    ranking_list[entity] = 1 / (i + 1) if entity not in ranking_list else ranking_list[entity] + 1 / (
                                i + 1)  # ranking_list[entity] / (i+1)
            else:
                ranking_list[res_entity] = 1 / (i + 1) if res_entity not in ranking_list else ranking_list[
                                                                                                  res_entity] + + 1 / (
                                                                                                          i + 1)  # ranking_list[res_entity] / (i+1)

        return max(ranking_list, key=ranking_list.get)

    def calc_distance(self, vector_a, vector_b):
        # Similaridade do cosseno
        # return tf.keras.losses.CosineSimilarity(axis=0)(vector_a, vector_b).numpy()

        # Distância Euclidiana
        return tf.norm(vector_a - vector_b, ord='euclidean')

    def get_embeddings(self, embeddings_file, embeddings_of):
        embeddings_list = []

        with open(embeddings_file, encoding='utf8') as embedding_file:
            lines = embedding_file.readlines()

            for i in range(len(lines)):

                if embeddings_of == Evaluate.ONLY_URIS:
                    if not (lines[i].startswith("https:") or lines[i].startswith("http:")):
                        continue
                elif embeddings_of == Evaluate.ONLY_LITERALS:
                    if (lines[i].startswith("https:") or lines[i].startswith("http:")):
                        continue

                line_splitted = lines[i].split('|||')
                try:
                    embeddings_list.append(
                        (line_splitted[0], tf.constant([float(x) for x in line_splitted[1].split(' ')])))
                except Exception as e:
                    print(e)
                    logging.info(f"Message: {e}")

        return embeddings_list

    def get_literals_relations(self, literals_relations_file):
        literals_relations = {}

        with open(literals_relations_file, encoding='utf8') as literals_relations_file:
            lines = literals_relations_file.readlines()

            for i in range(len(lines)):

                line_splitted = lines[i].split('|||')

                if line_splitted[0] in literals_relations:
                    for entity in line_splitted[1:]:
                        literals_relations[line_splitted[0]].append(entity.strip())
                else:
                    literals_relations[line_splitted[0]] = []
                    for entity in line_splitted[1:]:
                        literals_relations[line_splitted[0]].append(entity.strip())

        return literals_relations
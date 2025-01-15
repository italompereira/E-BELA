#import builtins
#import logging
#import math
import os
#import pickle
import json
import time
import re
import random

import tensorflow as tf
#from pyspark.sql import functions as F
from pyspark.sql.functions import *
#from pyspark.sql.types import StructType, StructField, StringType
#from pyspark.sql.window import Window as W
from sklearn.neighbors import KDTree

from Encoders.encoder import Encoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import spacy

from Encoders.encoder_sentence_transformer import EncoderTransformer
from Encoders.encoder_universal_sentence_encoder import EncoderUSE


class Evaluate:

    ONLY_URIS = 0
    ONLY_LITERALS = 1
    ALL = 2
    SEED = 448

    PATHS = {
        'lcquad': './Datasets/LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json',
        'aida': './Datasets/Conll/aida-yago2-dataset/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv',
        'aidab': './Datasets/Conll/aida-yago2-dataset/aida-yago2-dataset/aida_testB.tsv',
        'ace2004': './Datasets/ed_data/entity_disambiguation/ace2004.conll',
        'aquaint': './Datasets/ed_data/entity_disambiguation/aquaint.conll',
        'msnbc': './Datasets/ed_data/entity_disambiguation/msnbc.conll',
        'qald-7': './Datasets/qald/qald-7-annotated.json',
        'qald-8': './Datasets/qald/qald-8-annotated.json',
        'qald-9': './Datasets/qald/qald-9-annotated.json',
    }

    def __init__(self, spark_data, encoder, data_base, context_with_mention_avg, dataset, config_experiment):
        self.spark_data = spark_data
        self.encoder = encoder
        self.dataBase = data_base
        self.context_with_mention_avg = context_with_mention_avg
        self.config_experiment = config_experiment
        self.dataset = dataset

        if self.dataset in ['lcquad','qald-7']:
            self.ground_truth = self.load_lcquad_grounth_truth(Evaluate.PATHS[self.dataset])
        else:
            self.ground_truth = self.load_conll_grounth_truth(Evaluate.PATHS[self.dataset])

    def __call__(self, *args, **kwargs):
        print('Evaluating embeddings.')
        self.evaluate()

    def get_mention_context(self, mention, window_radius, corpus):
        """ Returns a list containing the window_size amount of words to the left
        and to the right of word_index
        """
        corpus = [token.text for token in corpus]
        mention = [token.text for token in mention]
        max_length = len(corpus)

        #indexes = [corpus.index(token) for token in mention]
        indexes = [i for token_m in mention for i in range(len(corpus)) if token_m == corpus[i]]

        if len(indexes) > 1:
            left = indexes[0]
            right = indexes[-1]
        else:
            left = indexes[-1]
            right = indexes[-1]

        left_border = left - window_radius
        left_border = 0 if left_border < 0 else left_border
        right_border = right + 1 + window_radius
        right_border = max_length if right_border > max_length else right_border
        return " ".join(corpus[left_border:left] + mention + corpus[right+1: right_border])

    def preprocess_literal(self, object):
        # Preprocess the literal
        literal = re.sub('\r?@en .', '', object)
        literal = re.sub(r'[^a-zA-Z0-9()\'\"-.,:<>?!@$%&|\s]+', '', literal).strip('[" ]')  # remove special characters
        # literal = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", literal)  # CamelCase split
        return literal

    def evaluate(self):


        #transformer = EncoderUSE(self.spark_data)
        transformer = self.encoder.transformer #EncoderTransformer(self.spark_data, model="ST", weighted=False, strategy="AVG", top_n=2, top_n_way='GROUPED', config_experiment='')
        if type(self.ground_truth) is tuple:
            ground_truth = self.ground_truth[0]
            documents = self.ground_truth[1]
        else:
            ground_truth = self.ground_truth
            documents = None

        entities_mapping = [(item['label'], item['uri'], i) for i in range(len(ground_truth)) for item in ground_truth[i]['entity mapping']]
        random.seed(Evaluate.SEED)
        random.shuffle(entities_mapping)
        #entities_mapping = [(item['label'], item['uri'], i) for i in range(len(ground_truth)) for item in ground_truth[i]['predicate mapping'] if 'label' in item]

        print("Embedding " + str(len(entities_mapping)) + " entities from Ground Truth.")

        vectors = transformer.get_embedding([str.lower(self.preprocess_literal(i[0])) for i in entities_mapping]).tolist()

        Y_pred = []
        Y_pred_disambiguated = []
        i = 0
        number_instance_sample = 0

        indexes_not_in_database = []
        file_number = 1
        if not os.path.exists(f'./EvaluateLogs/errors_{self.config_experiment}_{file_number}.txt'):
        # while os.path.exists(f'./EvaluateLogs/errors_{self.config_experiment}_{file_number}.txt'):
        #     file_number += 1
            error_file = open(f'./EvaluateLogs/errors_{self.config_experiment}_{file_number}.txt', 'w', encoding='utf-8')
            error_file.write(f'{self.encoder.config_experiment}\n')

            result_file = open(f'./EvaluateLogs/results_{self.config_experiment}_{file_number}.txt', 'w', encoding='utf-8')
            result_file.write(f'{self.config_experiment}\n')

            disambiguation_file = open(f'./EvaluateLogs/disambiguation_{self.config_experiment}_{file_number}.txt', 'w', encoding='utf-8')
            disambiguation_file.write(f'{self.config_experiment}\n')

            log_file = open(f'./EvaluateLogs/log_{self.config_experiment}_{file_number}.txt', 'w', encoding='utf-8')
            log_file.write(f'{self.config_experiment}\n')
            log_file.write(f'Index|y_true|y_pred|Status\n')
        else:
            error_file = open(f'./EvaluateLogs/errors_{self.config_experiment}_{file_number}.txt', 'a+', encoding='utf-8')
            result_file = open(f'./EvaluateLogs/results_{self.config_experiment}_{file_number}.txt', 'a+', encoding='utf-8')
            disambiguation_file = open(f'./EvaluateLogs/disambiguation_{self.config_experiment}_{file_number}.txt', 'a+', encoding='utf-8')
            log_file = open(f'./EvaluateLogs/log_{self.config_experiment}_{file_number}.txt', 'r+', encoding='utf-8')

            lines = log_file.readlines()[2:]
            indexes_not_in_database = [int(line.split('|')[0][6:]) for line in lines if 'Empty|Not present at database' in line]
            number_of_not_in_database = len(indexes_not_in_database)
            number_of_indexes = int(lines[-1].split('|')[0][6:])+1
            i = number_of_indexes
            number_instance_sample = number_of_indexes - number_of_not_in_database

            indexes_disambiguated = [int(line.split('|')[0][6:]) for line in lines if 'Disambiguated' in line]
            Y_pred = [line.split('|')[2].replace('\n', '') for line in lines]
            Y_pred_disambiguated = [line.split('|')[2].replace('\n', '') for line in lines]
            for index_disambiguated in indexes_disambiguated:
                Y_pred_disambiguated[index_disambiguated] = lines[index_disambiguated].split('|')[1].replace('\n', '')

            for index_not_in_database in sorted(indexes_not_in_database, reverse=True):
                del Y_pred[index_not_in_database]
                del Y_pred_disambiguated[index_not_in_database]


        nlp = spacy.load("en_core_web_sm")
        last_mention = ''
        if hasattr(self.encoder.transformer.embedding_model, 'max_seq_length'):
            max_seq_length = self.encoder.transformer.embedding_model.max_seq_length
        else:
            max_seq_length = None

        sample_size = 370 if len(vectors) > 370  else  len(vectors)
        while number_instance_sample < len(vectors[:sample_size]):
            print(str(i) + " vectors processed of " + str(sample_size))  # , end=".\t")
            try:
                mention = entities_mapping[i][0]
                y_true = entities_mapping[i][1]
                doc = nlp(ground_truth[entities_mapping[i][2]]['question'])
                #mention_doc = nlp(mention)
                #context = self.get_mention_context(mention_doc, 15, doc)
                if documents is not None:
                    document = self.preprocess_literal(documents[ground_truth[entities_mapping[i][2]]['document_key']])
                    ls_literals = []
                    if max_seq_length is not None and len(document) > max_seq_length:
                        text_sents = ''
                        for sent in [str(sent) for sent in nlp(document).sents]:
                            # if len(text_sents + sent) <= max_seq_length or text_sents == '':
                            #     text_sents += sent + ' '
                            #     continue
                            # ls_literals.append(text_sents)
                            # text_sents = sent
                            if len(str(sent)) < 3 or str.lower(mention) not in str.lower(sent):
                                continue
                            ls_literals.append(sent)

                        # if text_sents != '':
                        #     ls_literals.append(text_sents)
                    else:
                        ls_literals.append(document)
                    context = ls_literals
                else:
                    context = [self.preprocess_literal(ground_truth[entities_mapping[i][2]]['question'])] #self.get_mention_context(mention_doc, 15, doc)

                #context_vector = tf.reduce_mean(tf.stack(transformer.get_embedding(context).tolist(), axis=1), 1).numpy().tolist()
                context_vector = transformer.get_embedding(context).tolist()

                if self.context_with_mention_avg:
                    context_vector = tf.reduce_mean(tf.stack([vectors[i]] + context_vector, axis=1), 1)

                if mention != last_mention:
                    candidates = self.dataBase.get_top_k(vectors[i], y_true, 100)
                    time.sleep(5)
                last_mention = mention
                if candidates is None or len(candidates) == 0:
                    print('')
                    print(f'The entity {entities_mapping[i][1]} is not present at database.')
                    log_file.write(f'Index:{i}|{y_true}|Empty|Not present at database\n')

                    indexes_not_in_database.append(i)
                    i += 1
                    last_mention = ''
                    continue

                candidates_embedding = tf.constant([[float(x) for x in candidate[3].strip('][').split(',')] for candidate in candidates])
                distance = self.calc_distance(context_vector, candidates_embedding, axis=1)
                indexes = tf.math.top_k(tf.negative(distance), k=len(candidates))[1].numpy()

                repeated_distances = []
                for indexes_i in range(len(indexes)):
                    actual_distance = distance[indexes[indexes_i]].numpy()
                    if actual_distance == distance[indexes[indexes_i+1]].numpy():
                        if candidates[indexes[indexes_i]][2] == 'E':
                            repeated_distances.append(candidates[indexes[indexes_i]])
                        continue
                    break

                if len(repeated_distances) > 0:
                    Y_pred.append(candidates[indexes[0]][0])
                    Y_pred_disambiguated.append(candidates[indexes[0]][0])
                else:
                    Y_pred.append(candidates[indexes[0]][0])
                    Y_pred_disambiguated.append(candidates[indexes[0]][0])
                #Y_pred.append(self.get_similars_to(candidates, 10))
                # print if y_pred different to y_true

                if str.lower(Y_pred[-1]) != str.lower(y_true):
                    df_disambiguation = self.spark_data.df_disambiguation.filter(col('s') == y_true).select('o').withColumnRenamed("o", "entity")
                    df_disambiguation = df_disambiguation.union(self.spark_data.df_disambiguation.filter(col('o') == y_true).select('s'))
                    df_disambiguation_aux = df_disambiguation.union(
                        self.spark_data.df_disambiguation.filter(lower(col('s')).isin([str.lower(row.asDict()['entity']) for row in df_disambiguation.collect()]))
                        .select('o'))
                    df_disambiguation_aux = df_disambiguation_aux.union(
                        self.spark_data.df_disambiguation.filter(lower(col('o')).isin([str.lower(row.asDict()['entity']) for row in df_disambiguation.collect()]))
                        .select('s'))
                    rows = [str.lower(row.asDict()['entity']) for row in df_disambiguation_aux.collect()]
                    if str.lower(Y_pred[-1]) in rows:
                        log_file.write(f'Index:{i}|{y_true}|{Y_pred[-1]}|Disambiguated\n')
                        disambiguation_file.write(f'Index:{i}|{y_true}|{Y_pred[-1]}|Disambiguated\n')
                        Y_pred_disambiguated[-1] = y_true
                        i += 1
                        number_instance_sample += 1
                        continue

                    message = (f'\n\ni: {i}')
                    message += (f'\nMention: {mention}')
                    message += (f'\nContext: {context}')
                    message += (f'\nAll context: {doc.text}')
                    message += (f'\ny_true: {y_true}, y_pred: {Y_pred[-1]}')
                    print(message)
                    message += ("\nCandidates:\n" + "\n".join(str(line) for line in [(candidates[index][0], candidates[index][1], distance[index]) for index in indexes[:25]]))

                    error_file.write(message)
                    log_file.write(f'Index:{i}|{y_true}|{Y_pred[-1]}|Error\n')
                    i += 1
                    number_instance_sample += 1
                    continue
            except Exception as ex:
                print(f'i: {i}. Exception: {ex}')
                i += 1
                number_instance_sample += 1
                time.sleep(5)
                continue

            log_file.write(f'Index:{i}|{y_true}|{Y_pred[-1]}|Correct\n')
            i += 1
            number_instance_sample += 1

        # # Write the result of evaluation
        # file_number = 0
        # while os.path.exists(f'./EvaluateLogs/results_{self.config_experiment}_{file_number}.txt'):
        #     file_number += 1
        # results_file = open(f'./EvaluateLogs/results_{self.config_experiment}_{file_number}.txt', 'w', encoding='utf-8')
        # results_file.write(f'{self.config_experiment}\n')

        Y_pred_ = list(map(str.lower, Y_pred))[:i]
        Y_pred_disambiguated_ = list(map(str.lower, Y_pred_disambiguated))[:i]
        #Y_true = list(map(str.lower, [item['uri'] for i in range(len(ground_truth)) for item in ground_truth[i]['entity mapping']]))
        Y_true = [str.lower(i[1]) for i in entities_mapping]
        for index_not_in_database in sorted(indexes_not_in_database, reverse=True):
            del Y_true[index_not_in_database]
        Y_true = Y_true[:len(Y_pred_)]

        result = f'Evaluate {i} instances.\n\n'

        result += 'Not Disambiguated: \n'

        result += 'Accuracy: \n'
        result += f"Accuracy: {accuracy_score(Y_true, Y_pred_)}\n"

        result += 'Precision: \n'
        result += f"Macro precision: {precision_score(Y_true, Y_pred_, average='macro')}\n"
        result += f"Micro precision:  {precision_score(Y_true, Y_pred_, average='micro')}\n"

        result += 'Recall: \n'
        result += f"Macro recall: {recall_score(Y_true, Y_pred_, average='macro')}\n"
        result += f"Micro recall:  {recall_score(Y_true, Y_pred_, average='micro')}\n"

        result += 'F1: \n'
        result += f"Macro F1: {f1_score(Y_true, Y_pred_, average='macro')}\n"
        result += f"Micro F1:  {f1_score(Y_true, Y_pred_, average='micro')}\n"

        result += 'Disambiguated: \n'

        result += 'Accuracy: \n'
        result += f"Accuracy: {accuracy_score(Y_true, Y_pred_disambiguated_)}\n"

        result += 'Precision: \n'
        result += f"Macro precision: {precision_score(Y_true, Y_pred_disambiguated_, average='macro')}\n"
        result += f"Micro precision:  {precision_score(Y_true, Y_pred_disambiguated_, average='micro')}\n"

        result += 'Recall: \n'
        result += f"Macro recall: {recall_score(Y_true, Y_pred_disambiguated_, average='macro')}\n"
        result += f"Micro recall:  {recall_score(Y_true, Y_pred_disambiguated_, average='micro')}\n"

        result += 'F1: \n'
        result += f"Macro F1: {f1_score(Y_true, Y_pred_disambiguated_, average='macro')}\n"
        result += f"Micro F1:  {f1_score(Y_true, Y_pred_disambiguated_, average='micro')}\n"

        result += f'Entities not present in database: {len(indexes_not_in_database)}\n'
        for index in indexes_not_in_database:
            result += f"{entities_mapping[index][0]} - {entities_mapping[index][1]} \n"
        result_file.write(result)

        error_file.close()
        result_file.close()
        disambiguation_file.close()
        log_file.close()



        #print(sum(1 for x, y in zip(y_true, y_pred) if x == y) / len(y_true))

    # def set_centroid(self, row, full_centroids, s, number_rows_per_centroid, centroids, n_clusters):
    #
    #     distance = self.calc_distance(tf.constant(row.asDict()['embedding']), centroids, axis=1)
    #     id = row.asDict()['id']
    #     centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_clusters)[1].numpy()
    #     for centroid_index in centroid_indexes:
    #         # count += 1
    #         if full_centroids[centroid_index] < number_rows_per_centroid:  # or s.count(centroid_index) <= number_rows_per_centroid:
    #
    #             s[int(id) - 1] = centroid_index
    #             full_centroids[centroid_index] += 1
    #
    #             break
    #
    # def generate_blocking(self):
    #     print("Blocking")
    #     start = time.time()
    #
    #     df_embeddings_block = None
    #     centroids = None
    #     schema_df_s = StructType([
    #         StructField('id', StringType(), True),
    #         StructField('centroid', StringType(), True),
    #     ])
    #
    #     n_clusters = 250
    #
    #     if not os.path.exists(self.spark_data.PATH_EMBEDDINGS + "embeddings_with_blocks"):
    #
    #         if not os.path.exists(self.spark_data.PATH_AUX + "embeddings_aux"):
    #
    #             df_embeddings = self.spark_data.load_Embeddings().sort(col('s'))  # .limit(500)
    #
    #             # Add a column id to dataframe of embeddings
    #             w = W().partitionBy(F.lit(0)).orderBy(F.monotonically_increasing_id())
    #             df_embeddings_aux = df_embeddings.withColumn('id', F.row_number().over(w))
    #             df_embeddings_aux.write.json(self.spark_data.PATH_AUX + "embeddings_aux")
    #
    #         changed = True
    #         iteration = 0
    #         schema_df_embeddings_aux = StructType(list(set(self.spark_data.schema_df_embeddings.fields + [StructField('id', StringType(), True)])))
    #         df_embeddings_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings_aux).json(self.spark_data.PATH_AUX + "embeddings_aux").sort(col('id').cast("int"))#.limit(500)
    #
    #         data_len = df_embeddings_aux.count()
    #
    #         # Initialize the centroids
    #         fraction_size = 1 / (data_len / (n_clusters + 20))
    #         centroids = [tf.constant(row.asDict()['embedding']) for row in df_embeddings_aux.sample(fraction=fraction_size, seed=3).limit(n_clusters).toLocalIterator()]
    #         # centroids = [tf.constant(row[3]) for row in df_embeddings.sampleBy("typeObject", fractions={'E': fraction_size, 'L': fraction_size}, seed=10).sort('s').limit(n_clusters).toLocalIterator()]
    #         number_rows_per_centroid = math.ceil(data_len / n_clusters)
    #
    #         while changed and iteration < 10:
    #             print("Iteration " + str(iteration))
    #
    #             df_iterator = df_embeddings_aux.toLocalIterator()
    #             iteration += 1
    #             changed = False
    #
    #             s = [-1] * data_len
    #
    #             # appends elements to centroids
    #             full_centroids = [0] * n_clusters
    #             count = 0
    #
    #             for row in df_iterator:
    #                 distance = self.calc_distance(tf.constant(row.asDict()['embedding']), centroids, axis=1)
    #                 id = int(row.asDict()['id'])
    #                 if count % 100000 == 0:
    #                     print(str(id) + "\t vectors processed of " + str(data_len), end=".\n")
    #                 centroid_indexes = tf.math.top_k(tf.negative(distance), k=n_clusters)[1].numpy()
    #                 for centroid_index in centroid_indexes:
    #                     if full_centroids[centroid_index] < number_rows_per_centroid:  # or s.count(centroid_index) <= number_rows_per_centroid:
    #
    #                         s[id - 1] = centroid_index
    #                         full_centroids[centroid_index] += 1
    #                         break
    #                 count += 1
    #             # Builds a dataframe with centroids ids and joins with embeddings dataframe
    #             df_s = self.spark_data.sparkSession.createDataFrame([(i + 1, s[i].item()) for i in range(len(s))], schema_df_s)
    #             df_embeddings_block = df_embeddings_aux.join(df_s, df_embeddings_aux.id == df_s.id).drop(df_s.id).sort('centroid')
    #
    #         end = time.time()
    #         print(f'End time: {print(end - start)}')
    #
    #         df_embeddings_block.write.json(self.spark_data.PATH_EMBEDDINGS + "embeddings_with_blocks")
    #         #shutil.rmtree(self.data.PATH_EMBEDDINGS + "/embeddings_aux")
    #         with open(self.spark_data.PATH_EMBEDDINGS + '/centroids_trees.pickle', 'wb') as handle:
    #             pickle.dump(centroids, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #     if centroids is None:
    #         with open(self.spark_data.PATH_EMBEDDINGS + '/centroids_trees.pickle', 'rb') as handle:
    #             centroids = pickle.load(handle)
    #
    #     schema_df_embeddings_block = StructType(list(set(self.spark_data.schema_df_embeddings.fields + schema_df_s.fields)))
    #     df_embeddings_block = self.spark_data.sparkSession.read.schema(schema_df_embeddings_block).json(self.spark_data.PATH_EMBEDDINGS + "embeddings_with_blocks")
    #
    #
    #     if not os.path.exists(self.spark_data.PATH_TREES):
    #         print(f'Generating Trees')
    #         os.mkdir(self.spark_data.PATH_TREES)
    #
    #         for centroid in range(len(centroids)):
    #             start = time.time()
    #             centroid_data = df_embeddings_block.filter(F.col('centroid') == centroid).collect()
    #
    #             centroid_data_embeddings, centroid_data_indexes = map(list, zip(*[(row.asDict()['embedding'], (row.asDict()['id'], row.asDict()['s'], row.asDict()['o'])) for row in centroid_data]))
    #
    #             tree = KDTree(centroid_data_embeddings, leaf_size=2)
    #
    #             with open(self.spark_data.PATH_TREES + '/' + str(centroid) + '_tree' + '.pickle', 'wb') as handle:
    #                 pickle.dump(tree, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #             with open(self.spark_data.PATH_TREES + '/' + str(centroid) + '_indexes' + '.pickle', 'wb') as handle:
    #                 pickle.dump(centroid_data_indexes, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    #             end = time.time()
    #             print(f'Saving trees: {centroid}. End time: {(end - start)}')
    #
    #
    #     return centroids
    #
    #
    #
    # def load_tree(self, centroid):
    #     with open(self.spark_data.PATH_TREES + '/' + str(centroid) + '_tree' + '.pickle', 'rb') as handle:
    #         tree = pickle.load(handle)
    #
    #     with open(self.spark_data.PATH_TREES + '/' + str(centroid) + '_indexes' + '.pickle', 'rb') as handle:
    #         centroid_data = pickle.load(handle)
    #
    #     return tree, centroid_data
    # def generate_similarity_matrix(self, vectors):
    #     print("Generating similarity matrix")
    #
    #     start = time.time()
    #     centroids = self.generate_blocking()
    #     end = time.time()
    #     print(end - start)
    #     top_k = {}
    #     top_k_distance = {}
    #     for i in range(len(vectors)):
    #         if i % 100 == 0:
    #             print(str(i) + " vectors processed of " + str(len(vectors)), end=".\n")
    #             print(end - start)
    #         start = time.time()
    #         distance_centroids = self.calc_distance(vectors[i], centroids, axis=1)
    #         centroid_indexes = tf.math.top_k(tf.negative(distance_centroids), k=10)[1].numpy()
    #         top_k[i] = []
    #         top_k_distance[i] = []
    #
    #         # Find nearest elements in trees
    #         for centroid_index in centroid_indexes:
    #             tree, indexes = self.load_tree(centroid_index)
    #             distance_array, indexes_distance = tree.query([vectors[i]], k=30)
    #             top_k_size = 100 if len(distance_array) >= 100 else len(distance_array[0])
    #             top_k[i] += [indexes[index_distance][1] for index_distance in indexes_distance[0]][:top_k_size]
    #             top_k_distance[i] += list(distance_array[0])
    #
    #         indexes = tf.math.top_k(tf.negative(top_k_distance[i]), k=len(top_k_distance[i]))[1].numpy()
    #         top_k[i] = [top_k[i][index] for index in indexes]
    #
    #         end = time.time()
    #
    #     return top_k
    #
    #
    # def get_similars_to(self, top_k, k):
    #     ranking = top_k #[i]
    #
    #     #ranking_list_aux = top_k[i]
    #     ranking_list_aux = []
    #     order = 0
    #     ranking_list_aux.append((ranking[0][0], order))
    #     for j in range(1,len(ranking)):
    #         if not ranking[j][1] == ranking[j - 1][1]:
    #             order += 1
    #         ranking_list_aux.append((ranking[j][0], order))
    #
    #
    #
    #
    #     ranking_list = {}
    #     #order = 0
    #     #factor = ranking_list_aux[0][1]
    #     for i in range(len(ranking_list_aux)):
    #         res_entity = str.lower(ranking_list_aux[i][0])
    #         # if not (res_entity.startswith("https:") or res_entity.startswith("http:")):
    #         #     for entity in self.literal_relations[res_entity]:
    #         #         ranking_list[entity] = 1 / (i + 1) if entity not in ranking_list else ranking_list[entity] + 1 / (i + 1)
    #         # else:
    #         factor = ranking_list_aux[i][1]
    #         ranking_list[res_entity] = 1 / (factor + 1) if res_entity not in ranking_list else ranking_list[res_entity] + 1 / (factor + 1)
    #
    #     return builtins.max(ranking_list, key=ranking_list.get)

    def calc_distance(self, vector_a, vector_b, axis=None):
        # Similaridade do cosseno
        # return tf.keras.losses.CosineSimilarity(axis=1, reduction='none')([vector_a], vector_b).numpy()

        # Distância Euclidiana
        return tf.norm(vector_a - vector_b, axis=axis, ord='euclidean')

        # Distância Euclidiana
        #return tf.matmul(vector_a,tf.transpose(vector_b))




    def load_lcquad_grounth_truth(self, path_dataset):
        #path_dataset = './Datasets/LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json'

        with open(path_dataset, encoding='utf8') as data_file:
            dataset = json.load(data_file)

        return [{'question':item['question'], 'entity mapping':item['entity mapping'], 'predicate mapping':item['predicate mapping']} for item in dataset]

    def load_conll_grounth_truth(self, path_dataset):
        # question = ''
        # entity_mapping = {}
        # predicate_mapping = {}

        #path_dataset = './Datasets/Conll/aida-yago2-dataset/aida-yago2-dataset/AIDA-YAGO2-dataset.tsv'

        with open(path_dataset, encoding='utf8') as data_file:
            dataset = data_file.read()

        questions = []
        documents = dataset.split('-DOCSTART-')[1:]

        documents_dict = {}

        for document in documents:
            lines = document.split('\n')
            document_key = lines[0]
            document_sentences = ''

            questions_lines = lines[1:]

            question = ''
            entity_mapping = []
            predicate_mapping = []
            for line in questions_lines:
                if line != '':
                    split_line = line.split('\t')
                    question += split_line[0] + ' '
                    document_sentences += split_line[0] + ' '

                    if any(re.findall(r'http://en.wikipedia.org/wiki/|en.wikipedia.org/wiki/',line)):
                        if 'http://' not in line:
                            split_line[4] = 'http://' + split_line[4]

                    #if "http://en.wikipedia.org/wiki/" in line:
                        try:
                            entity_mapping.append({'label':split_line[2], 'uri': split_line[4].replace('en.wikipedia.org/wiki','dbpedia.org/resource')})
                        except:
                            print(split_line)
                else:
                    entity_mapping = list({v['label']: v for v in entity_mapping}.values())
                    questions.append({'question': question, 'entity mapping': entity_mapping, 'predicate mapping': predicate_mapping, 'document_key': document_key})
                    question = ''
                    entity_mapping = []
                    predicate_mapping = []

            documents_dict[document_key] = document_sentences

        return (questions, documents_dict)
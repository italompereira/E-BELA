import os
# import logging
import tensorflow as tf
import tensorflow_hub as hub
import re
import time

from PIL.ImageShow import show
from _distutils_hack import shim
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType, LongType, IntegerType
from pyspark.sql.window import Window


from Dumps.spark_data import SparkData
from sentence_transformers import SentenceTransformer
import spacy

class EncoderTransformer():

    def __init__(self, spark_data):
        print("Downloading model")
        #'SeyedAli/Multilingual-Text-Semantic-Search-Siamese-BERT-V1'
        #'sentence-transformers/all-mpnet-base-v2'
        #'nreimers/MiniLM-L6-H384-uncased'
        #'msmarco-distilroberta-base-v3'
        #'msmarco-distilbert-base-v4'
        path = 'sentence-transformers/all-mpnet-base-v2'
        if not os.path.exists('./'+path):
            self.embedding_model = SentenceTransformer(path)
            self.embedding_model.save('./'+path)
        else:
            self.embedding_model = SentenceTransformer('./'+path)

        #self.embedding_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
        self.spark_data = spark_data

    def preprocess_literal(self, object):
        # Preprocess the literal
        literal = re.sub('\r?@en .', '', object)
        literal = re.sub(r'[^a-zA-Z0-9()\'\".,:<>?!@$%&|\s]+', '', literal).strip('[" ]')   # remove special characters
        # literal = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", literal)  # CamelCase split
        return literal

    def fit(self, strategy, weighted = False, top_n = None):
        nlp = spacy.load("en_core_web_sm")
        if not os.path.exists(SparkData.PATH_EMBEDDINGS + 'done.txt'):
            schema_df_embeddings = self.spark_data.schema_df_embeddings
            schema_df_embeddings_full = self.spark_data.schema_df_embeddings_full

            # Store representations of literals, the first representation of entities and representations of entities
            df_emb_entities_temp = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)
            df_emb_entities = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)

            batch = 0
            batch_size = 10000
            data_frame_size = 1000000

            ########## Encoding literals ##########
            print('Encoding Literals')
            start = time.time()
            path_embedding_literals = SparkData.PATH_EMBEDDINGS_LITERALS
            path_embedding_entities_temp = SparkData.PATH_EMBEDDINGS_ENTITIES_TEMP

            # Counts how many occurrences of a literal to weight this in the encoder process
            df_literals = self.spark_data.df_literals.join(
                self.spark_data.df_literals.groupBy(lower("o")).count().withColumnRenamed("lower(o)", "o_")
                , (lower(col('o')) == lower(col('o_')))
            ).drop('o_')

            if strategy == 'average':

                df_emb_literals = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings_full)
                if not os.path.exists(path_embedding_entities_temp):
                    if os.path.exists(path_embedding_literals):
                        # Check if there are some literals not embedded
                        print(f"Loading existing embeddings from {path_embedding_literals}")
                        df_emb_literals_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings_full).json(path_embedding_literals).distinct()
                        temp = df_literals.join(
                            df_emb_literals_aux.select('s', 'p', 'o')
                            .withColumnRenamed("s", "s_")
                            .withColumnRenamed("p", "p_")
                            .withColumnRenamed("o","o_"),
                            (col('s') == col('s_')) & (col('p') == col('p_')) & (col('o') == col('o_')), 'left_anti')
                    else:
                        temp = df_literals #.sort('s').limit(500)

                    index_row = 0
                    index_row_dataframe = 0
                    size = 0 #temp.count()
                    if size > 0:
                        df_iterator = temp.sort('s','count').toLocalIterator()
                        # Parallel(n_jobs=10)(delayed(print)(row.asDict()['s']) for row in df_iterator)
                        ls_emb_aux = []
                        ls_literals = []
                        ls_weights_literals = []
                        for row in df_iterator:
                            index_row += 1
                            index_row_dataframe += 1

                            # Gets the subject and object
                            row_as_dict = row.asDict()
                            subject = row_as_dict['s']
                            predicate = row_as_dict['p']
                            object = row_as_dict['o']
                            count = row_as_dict['count']

                            # Preprocess the literal
                            literal = self.preprocess_literal(object)
                            if literal == '':
                                continue

                            max_seq_length = self.embedding_model.max_seq_length

                            if len(literal) > max_seq_length:
                                text_sents = ''
                                for sent in [str(sent) for sent in nlp(literal).sents]:
                                    if len(text_sents + sent) <= max_seq_length or text_sents == '':
                                        text_sents += sent + ' '
                                        continue
                                    ls_emb_aux.append((subject, predicate, text_sents, 'L'))
                                    ls_literals.append(text_sents)
                                    ls_weights_literals.append([count])
                                    text_sents = sent

                                if text_sents != '':
                                    ls_emb_aux.append((subject, predicate, text_sents, 'L'))
                                    ls_literals.append(text_sents)
                                    ls_weights_literals.append([count])

                            else:
                                ls_emb_aux.append((subject, predicate, literal, 'L'))
                                ls_literals.append(literal)
                                ls_weights_literals.append([count])

                            if batch == batch_size:
                                batch = 0
                                embeddings = self.get_embedding(ls_literals)#.numpy().tolist()
                                if weighted:
                                    ls_weights_literals = tf.broadcast_to(tf.constant(ls_weights_literals, dtype=float),(embeddings.shape))
                                    embeddings = (embeddings/ls_weights_literals).numpy().tolist()
                                else:
                                    embeddings = embeddings.tolist()

                                ls_emb_aux = [(ls_emb_aux[i][0], ls_emb_aux[i][1], ls_emb_aux[i][2], ls_emb_aux[i][3], embeddings[i])  for i in range(len(ls_emb_aux))]
                                df_emb_literals = df_emb_literals.union(
                                    self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings_full))
                                ls_emb_aux = []
                                ls_literals = []
                                ls_weights_literals = []

                                print(f'Encoding Literals: {index_row} of {size}')
                                end = time.time()
                                print(end - start)
                                start = time.time()

                                continue

                            #if index_row % 10000 == 0:
                            if index_row_dataframe > data_frame_size:
                                df_emb_literals.write.mode('append').json(path_embedding_literals + '_temp')
                                df_emb_literals = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings_full)
                                index_row_dataframe = 0
                            batch += 1
                        if df_emb_literals.count() > 0 or len(ls_emb_aux) > 0:
                            if len(ls_emb_aux) > 0:
                                embeddings = self.get_embedding(ls_literals)#.numpy().tolist()
                                if weighted:
                                    ls_weights_literals = tf.broadcast_to(tf.constant(ls_weights_literals, dtype=float), (embeddings.shape))
                                    embeddings = (embeddings / ls_weights_literals).numpy().tolist()
                                else:
                                    embeddings = embeddings.tolist()

                                ls_emb_aux = [(ls_emb_aux[i][0], ls_emb_aux[i][1], ls_emb_aux[i][2], ls_emb_aux[i][3], embeddings[i]) for i in range(len(ls_emb_aux))]

                            df_emb_literals.union(self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings_full)).write.mode('append').json(path_embedding_literals + '_temp')

                        end = time.time()
                        print(end - start)

                        # Adds columns count and top_n
                        df_emb_literals = self.spark_data.sparkSession.read.schema(schema_df_embeddings_full).json(path_embedding_literals + '_temp')

                        df_emb_literals = df_emb_literals.join(
                            df_emb_literals.groupBy(lower("o")).count().withColumnRenamed("lower(o)", "o_"),
                            (lower(col('o')) == lower(col('o_')))
                        ).drop('o_')
                        window_sub = Window.partitionBy(*["s"]).orderBy(["s","count"])
                        df_emb_literals = df_emb_literals.withColumn("top_n", row_number().over(window_sub))
                        df_emb_literals.write.mode('append').json(path_embedding_literals)
                        # os.remove(path_embedding_literals + '_temp')

                schema_ = StructType(schema_df_embeddings_full.fields + [StructField('count', LongType(), True), StructField('top_n', IntegerType(), True)])
                df_emb_literals = self.spark_data.sparkSession.read.schema(schema_).json(path_embedding_literals)

                if top_n is not None:
                    df_emb_literals = df_emb_literals.filter(col('top_n') <= top_n)


                ########## Gets the first representation of entities (average of literals) ##########
                print('Gets the first representation of entities (average of literals)')
                start = time.time()
                path_embedding_entities = SparkData.PATH_EMBEDDINGS_ENTITIES
                if not os.path.exists(path_embedding_entities):
                    if os.path.exists(path_embedding_entities_temp):
                        print(f"Loading existing embeddings from {path_embedding_entities_temp}")
                        df_emb_entities_temp_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities_temp)
                        temp = df_emb_literals.join(df_emb_entities_temp_aux.select('s').withColumnRenamed("s", "s_"),(col('s') == col('s_')), 'left_anti')
                    else:  # if not os.path.exists(path_embedding_entities_temp):
                        temp = df_emb_literals

                    index_row = 0
                    index_row_dataframe = 0
                    size = temp.count()
                    if size > 0:
                        df_iterator = temp.sort('s','count').toLocalIterator()
                        ls_emb_aux = []
                        tensors = []
                        current_row = next(iter(df_iterator))
                        label_or_name_embedding = []
                        top_n = 1
                        while True:
                            index_row += 1
                            index_row_dataframe += 1
                            try:
                                next_row = next(iter(df_iterator))
                            except:
                                tensors.append(tf.constant(current_row.asDict()['embedding']))
                                new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                                current_entity = current_row.asDict()['s']
                                ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))

                                df_emb_entities_temp = df_emb_entities_temp.union(
                                    self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                                ls_emb_aux = []
                                break
                            current_entity = current_row.asDict()['s']
                            current_predicate = current_row.asDict()['p']
                            current_object = current_row.asDict()['o']
                            next_entity = next_row.asDict()['s']
                            current_embedding = current_row.asDict()['embedding']

                            # Gets the label or name to merge with the representation of the string
                            if current_predicate in ['http://www.w3.org/2000/01/rdf-schema#label','http://xmlns.com/foaf/0.1/name'] and label_or_name_embedding == []:
                                label_or_name_embedding = current_embedding

                            tensors.append(tf.constant(current_embedding))

                            current_row = next_row
                            if current_entity == next_entity:
                                continue
                            else:
                                new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1).numpy().tolist()

                                if label_or_name_embedding != [] and len(tensors) > 1:
                                    #label_or_name_embedding = self.get_embedding([label_or_name]).tolist()[0]
                                    new_embedding = tf.reduce_mean(tf.stack([new_embedding, label_or_name_embedding], axis=1),1).numpy().tolist()

                                ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding))
                                tensors = []
                                label_or_name_embedding = []

                            if len(ls_emb_aux) == batch_size:
                                df_emb_entities_temp = df_emb_entities_temp.union(
                                    self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                                ls_emb_aux = []
                                print(f'Encoding Entities (Mean): {index_row} of {size}')
                                end = time.time()
                                print(end - start)
                                start = time.time()

                            if index_row_dataframe > data_frame_size:
                                df_emb_entities_temp.write.mode('append').json(path_embedding_entities_temp)
                                df_emb_entities_temp = self.spark_data.sparkSession.createDataFrame([],
                                                                                                    schema_df_embeddings)
                                index_row_dataframe = 0

                        if df_emb_entities_temp.count() > 0 or len(ls_emb_aux) > 0:
                            df_emb_entities_temp.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode(
                                'append').json(path_embedding_entities_temp)

                        end = time.time()
                        print(end - start)
                df_emb_entities_temp = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities_temp)


                ########## Encoding Entities ##########
                print('Encoding Entities')
                start = time.time()

                # Retrieve the embeddings of the child entities of each entity
                df_temp = (
                    (
                        (
                            df_emb_entities_temp
                            .select('s')
                            .withColumnRenamed('s', 's_')
                            .join(self.spark_data.df_entities, col('s') == col('s_'))
                            .select('s_', 'o')
                            .withColumnRenamed("o", "o_")
                        ).join(df_emb_entities_temp, col('o_') == col('s'))
                    ).select('s_', 'o_', 'embedding')
                    .union(df_emb_entities_temp.select('s', 'o', 'embedding'))
                    .sort('s_')
                )

                if os.path.exists(path_embedding_entities):
                    print(f"Loading existing embeddings from {path_embedding_entities}")
                    df_emb_entities_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities)
                    temp = df_temp.join(df_emb_entities_aux.select('s', 'o'), (col('s') == col('s_')), 'left_anti')
                else:  # if not os.path.exists(path_embedding_entities):
                    temp = df_temp

                index_row = 0
                index_row_dataframe = 0
                size = temp.count()
                if size > 0:
                    df_iterator = temp.toLocalIterator()
                    ls_emb_aux = []
                    tensors = []
                    current_row = next(iter(df_iterator))

                    while True:
                        index_row += 1
                        index_row_dataframe += 1
                        try:
                            next_row = next(iter(df_iterator))
                        except:
                            tensors.append(tf.constant(current_row.asDict()['embedding']))
                            new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                            current_entity = current_row.asDict()['s_']
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))

                            df_emb_entities = df_emb_entities.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                            break

                        current_entity = current_row.asDict()['s_']
                        next_entity = next_row.asDict()['s_']

                        tensors.append(tf.constant(current_row.asDict()['embedding']))

                        current_row = next_row
                        if current_entity == next_entity:
                            continue
                        else:
                            new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))
                            tensors = []

                        if len(ls_emb_aux) == batch_size:
                            df_emb_entities = df_emb_entities.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                        if index_row % 10000 == 0:
                            print(f'Encoding Entities: {index_row} of {size}')
                            end = time.time()
                            print(end - start)

                            if index_row_dataframe > data_frame_size:
                                df_emb_entities.write.mode('append').json(path_embedding_entities)
                                df_emb_entities = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)
                                index_row_dataframe = 0

                    end = time.time()
                    print(end - start)
                    if df_emb_entities.count() > 0 or len(ls_emb_aux) > 0:
                        df_emb_entities.union(
                            self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode(
                            'append').json(path_embedding_entities)

            else:

                df_emb_literals = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)
                if not os.path.exists(path_embedding_entities_temp):
                    if os.path.exists(path_embedding_literals):
                        # Check if there are some literals not embedded
                        print(f"Loading existing embeddings from {path_embedding_literals}")
                        df_emb_literals_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_literals).distinct()
                        #temp = (df_literals.join(df_emb_literals_aux.select('s', 'o').withColumnRenamed("s", "s_").withColumnRenamed("o","o_"), (col('o') == col('o_')) & (col('s') == col('s_')), 'left_anti'))
                        temp = (df_literals.join(df_emb_literals_aux.select('s').withColumnRenamed("s", "s_"),(col('s') == col('s_')), 'left_anti'))
                    else:
                        temp = df_literals #.sort('s').limit(5000)

                    index_row = 0
                    index_row_dataframe = 0
                    size = temp.count()
                    if size > 0:
                        df_iterator = temp.sort('s').toLocalIterator() #.groupby('s').agg(concat_ws('. ', collect_list(temp.o)).alias('o')).toLocalIterator()

                        ls_emb_aux = []
                        current_row = next(iter(df_iterator))
                        current_literal = ''
                        label_or_name = ''
                        while True:
                            index_row += 1
                            index_row_dataframe += 1
                            try:
                                next_row = next(iter(df_iterator))
                            except:
                                embedding = self.get_embedding(tf.constant([current_literal])).numpy().tolist()[0]
                                if label_or_name != '':
                                    label_or_name_embedding = self.get_embedding(tf.constant([label_or_name])).numpy().tolist()[0]
                                    embedding = tf.reduce_mean(tf.stack([embedding, label_or_name_embedding], axis=1),1).numpy().tolist()
                                ls_emb_aux.append((current_entity, current_literal, 'L', embedding))
                                break

                            # Gets the subject and object
                            #subject = (row.asDict())['s']
                            #object = (row.asDict())['o']

                            #embedding = self.get_embedding(tf.constant([literal])).numpy().tolist()[0]
                            #ls_emb_aux.append((subject, object, 'L', embedding))
                            current_row_dict = current_row.asDict()
                            current_entity = current_row_dict['s']
                            current_predicate = current_row_dict['p']
                            current_object = current_row_dict['o']
                            next_entity = next_row.asDict()['s']

                            # Gets the label or name to merge with the representation of the string
                            if label_or_name == '' and (current_predicate == 'http://www.w3.org/2000/01/rdf-schema#label' or current_predicate == 'http://xmlns.com/foaf/0.1/name'):
                                label_or_name = self.preprocess_literal(current_object)
                                current_literal += label_or_name
                            else:
                                current_literal += self.preprocess_literal(current_object) + '. '


                            # if current_literal == '':
                            #     continue

                            current_row = next_row
                            if current_entity == next_entity:
                                continue
                            else:
                                if label_or_name != '':
                                    embeddings = self.get_embedding(tf.constant([current_literal, label_or_name])).numpy().tolist()
                                    embedding = tf.reduce_mean(tf.stack(embeddings, axis=1), 1).numpy().tolist()
                                    label_or_name = ''
                                else:
                                    embedding = self.get_embedding(tf.constant([current_literal])).numpy().tolist()[0]
                                ls_emb_aux.append((current_entity, current_literal, 'L', embedding))
                                current_literal = ''
                                batch += 1

                            if batch == batch_size:
                                batch = 0
                                df_emb_literals = df_emb_literals.union(
                                    self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                                ls_emb_aux = []
                                continue

                            if index_row % 10000 == 0:
                                print(f'Encoding Literals: {index_row} of {size}')
                                end = time.time()
                                print(end - start)
                                start = time.time()
                                if index_row_dataframe > data_frame_size:
                                    df_emb_literals.write.mode('append').json(path_embedding_literals)
                                    df_emb_literals = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)
                                    index_row_dataframe = 0

                        if df_emb_literals.count() > 0 or len(ls_emb_aux) > 0:
                            df_emb_literals.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode(
                                'append').json(path_embedding_literals)

                            # self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings).write.mode('append').json(path_embedding_literals)
                        end = time.time()
                        print(end - start)
                df_emb_literals = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_literals)
                path_embedding_entities = SparkData.PATH_EMBEDDINGS_ENTITIES

                ########## Encoding Entities ##########
                print('Encoding Entities')
                start = time.time()

                # Retrieve the embeddings of the child entities of each entity
                df_emb_literals = df_emb_literals.withColumnRenamed("o", "l").withColumn("o", col('s'))
                df_temp = (
                    (
                        (
                            df_emb_literals
                            .select('s')
                            .withColumnRenamed('s', 's_')
                            .join(self.spark_data.df_entities, col('s') == col('s_'))
                            .select('s_', 'o')
                            .withColumnRenamed("o", "o_")
                        ).join(df_emb_literals, col('o_') == col('s'))
                    ).select('s_', 'o_', 'embedding')
                    .union(df_emb_literals.select('s', 'o', 'embedding'))
                    .sort('s_')
                )

                if os.path.exists(path_embedding_entities):
                    print(f"Loading existing embeddings from {path_embedding_entities}")
                    df_emb_entities_aux = self.spark_data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities)
                    temp = df_temp.join(df_emb_entities_aux.select('s', 'o'), (col('s') == col('s_')), 'left_anti')
                else:  # if not os.path.exists(path_embedding_entities):
                    temp = df_temp

                index_row = 0
                index_row_dataframe = 0
                size = temp.count()
                if size > 0:
                    df_iterator = temp.toLocalIterator()
                    ls_emb_aux = []
                    tensors = []
                    current_row = next(iter(df_iterator))

                    while True:
                        index_row += 1
                        index_row_dataframe += 1
                        try:
                            next_row = next(iter(df_iterator))
                        except:
                            tensors.append(tf.constant(current_row.asDict()['embedding']))
                            new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                            current_entity = current_row.asDict()['s_']
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))

                            df_emb_entities = df_emb_entities.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                            break
                        current_entity = current_row.asDict()['s_']
                        next_entity = next_row.asDict()['s_']

                        tensors.append(tf.constant(current_row.asDict()['embedding']))

                        current_row = next_row
                        if current_entity == next_entity:
                            continue
                        else:
                            new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))
                            tensors = []

                        if len(ls_emb_aux) == batch_size:
                            df_emb_entities = df_emb_entities.union(
                                self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                        if index_row % 10000 == 0:
                            print(f'Encoding Entities (Mean): {index_row} of {size}')
                            end = time.time()
                            print(end - start)

                            if index_row_dataframe > data_frame_size:
                                df_emb_entities.write.mode('append').json(path_embedding_entities)
                                df_emb_entities = self.spark_data.sparkSession.createDataFrame([], schema_df_embeddings)
                                index_row_dataframe = 0

                    end = time.time()
                    print(end - start)
                    if df_emb_entities.count() > 0 or len(ls_emb_aux) > 0:
                        df_emb_entities.union(
                            self.spark_data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode(
                            'append').json(path_embedding_entities)

            with open(SparkData.PATH_EMBEDDINGS + 'done.txt', 'w') as f:
                f.write('All entities and literals embedded!')
        # df_emb_entities = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities)
        # print(df_emb_literals.union(df_emb_entities).count())

        print('All entities and literals embedded!')
    def get_embedding(self, sentences):
        return self.embedding_model.encode(sentences, show_progress_bar=True)

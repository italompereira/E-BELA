import os
#import logging
import tensorflow as tf
import re
import time
from pyspark.sql.functions import *
from Dumps.spark_data import SparkData
from openai import OpenAI

#from pyspark.storagelevel import StorageLevel

class Encoder():

    def __init__(self, kg, data):
        import os
        os.environ["OPENAI_API_KEY"] = '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'

        encoder = OpenAI()
        OpenAI.api_key = os.getenv('OPENAI_API_KEY')
        self.embedding_model = encoder
        self.kg = kg
        self.data = data

    def embedding_literals(self, row, ):
        print('teste')

    def fit(self):

        if not os.path.exists(SparkData.PATH_EMBEDDINGS + 'done.txt'):
            schema_df_embeddings = self.data.schema_df_embeddings

            # Store representations of literals, the first representation of entities and representations of entities
            df_emb_literals = self.data.sparkSession.createDataFrame([], schema_df_embeddings)
            df_emb_entities_temp = self.data.sparkSession.createDataFrame([], schema_df_embeddings)
            df_emb_entities = self.data.sparkSession.createDataFrame([], schema_df_embeddings)

            batch = 0
            batch_size = 2000
            data_frame_size = 1000000

            ########## Encoding literals ##########
            print('Encoding Literals')
            start = time.time()
            path_embedding_literals = SparkData.PATH_EMBEDDINGS_LITERALS
            path_embedding_entities_temp = SparkData.PATH_EMBEDDINGS_ENTITIES_TEMP

            if not os.path.exists(path_embedding_entities_temp):
                if os.path.exists(path_embedding_literals):
                    # Check if there are some literals not embedded
                    print(f"Loading existing embeddings from {path_embedding_literals}")
                    df_emb_literals_aux = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_literals).distinct()
                    temp = self.data.df_literals.join(df_emb_literals_aux.select('s','o').withColumnRenamed("s","s_").withColumnRenamed("o","o_"), (col('o') == col('o_')) & (col('s') == col('s_')), 'left_anti')
                else:
                    temp = self.data.df_literals

                index_row = 0
                index_row_dataframe = 0
                size = temp.count()
                if size > 0:
                    df_iterator = temp.sort('s').toLocalIterator()

                    ls_emb_aux = []
                    for row in df_iterator:
                        index_row += 1
                        index_row_dataframe += 1

                        # Gets the subject and object
                        subject = (row.asDict())['s']
                        object = (row.asDict())['o']

                        # Preprocess the literal
                        literal = re.sub('\r?@en .', '', object)
                        #literal = re.sub(r'[^a-zA-Z0-9 ]+', '', literal).strip() #remove special characters
                        #literal = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", literal) #CamelCase split

                        if literal == '':
                            continue

                        embedding = self.get_embedding([literal])[0]
                        ls_emb_aux.append((subject, object, 'L', embedding))

                        if batch == batch_size:
                            batch = 0
                            df_emb_literals = df_emb_literals.union(
                                self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                            continue

                        if index_row % 10000 == 0:
                            print(f'Encoding Literals: {index_row} of {size}')
                            end = time.time()
                            print(end - start)

                            if index_row_dataframe > data_frame_size:
                                df_emb_literals.write.mode('append').json(path_embedding_literals)
                                df_emb_literals = self.data.sparkSession.createDataFrame([], schema_df_embeddings)
                                index_row_dataframe = 0
                        batch += 1
                    if df_emb_literals.count() > 0 or len(ls_emb_aux) > 0 :
                        df_emb_literals.union(
                            self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode('append').json(path_embedding_literals)

                        #self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings).write.mode('append').json(path_embedding_literals)
                    end = time.time()
                    print(end - start)
            df_emb_literals = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_literals)


            ########## Gets the first representation of entities (average of literals) ##########
            print('Gets the first representation of entities (average of literals)')
            start = time.time()
            path_embedding_entities = SparkData.PATH_EMBEDDINGS_ENTITIES
            if not os.path.exists(path_embedding_entities):
                if os.path.exists(path_embedding_entities_temp):
                    print(f"Loading existing embeddings from {path_embedding_entities_temp}")
                    df_emb_entities_temp_aux = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities_temp)
                    temp = df_emb_literals.join(df_emb_entities_temp_aux.select('s').withColumnRenamed("s","s_"), (col('s') == col('s_')), 'left_anti')


                else: #if not os.path.exists(path_embedding_entities_temp):
                    temp = df_emb_literals

                index_row = 0
                index_row_dataframe = 0
                size = temp.count()
                if size > 0:
                    df_iterator = temp.sort('s').toLocalIterator()
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
                            current_entity = current_row.asDict()['s']
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))

                            df_emb_entities_temp = df_emb_entities_temp.union(
                                self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []
                            break

                        current_entity = current_row.asDict()['s']
                        next_entity = next_row.asDict()['s']

                        tensors.append(tf.constant(current_row.asDict()['embedding']))

                        current_row = next_row
                        if current_entity == next_entity:
                            continue
                        else:
                            new_embedding = tf.reduce_mean(tf.stack(tensors, axis=1), 1)
                            ls_emb_aux.append((current_entity, current_entity, 'E', new_embedding.numpy().tolist()))
                            tensors = []

                        if len(ls_emb_aux) == batch_size:
                            df_emb_entities_temp = df_emb_entities_temp.union(
                                self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                            ls_emb_aux = []

                        if index_row % 10000 == 0:
                            print(f'Encoding Entities (Mean): {index_row} of {size}')
                            end = time.time()
                            print(end - start)

                            if index_row_dataframe > data_frame_size:
                                df_emb_entities_temp.write.mode('append').json(path_embedding_entities_temp)
                                df_emb_entities_temp = self.data.sparkSession.createDataFrame([], schema_df_embeddings)
                                index_row_dataframe = 0

                    if df_emb_entities_temp.count() > 0 or len(ls_emb_aux) > 0:
                        df_emb_entities_temp.union(
                            self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode('append').json(path_embedding_entities_temp)

                    end = time.time()
                    print(end - start)
            df_emb_entities_temp = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities_temp)

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
                        .join(self.data.df_entities, col('s') == col('s_'))
                        .select('s_', 'o')
                        .withColumnRenamed("o", "o_")
                    ).join(df_emb_entities_temp, col('o_') == col('s'))
                ).select('s_', 'o_', 'embedding')
                .union(df_emb_entities_temp.select('s', 'o', 'embedding'))
                .sort('s_')
            )

            if os.path.exists(path_embedding_entities):
                print(f"Loading existing embeddings from {path_embedding_entities}")
                df_emb_entities_aux = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities)
                temp = df_temp.join(df_emb_entities_aux.select('s', 'o'), (col('s') == col('s_')),'left_anti')

            else: #if not os.path.exists(path_embedding_entities):
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
                            self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
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
                            self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings))
                        ls_emb_aux = []
                    if index_row % 10000 == 0:
                        print(f'Encoding Entities (Mean): {index_row} of {size}')
                        end = time.time()
                        print(end - start)

                        if index_row_dataframe > data_frame_size:
                            df_emb_entities.write.mode('append').json(path_embedding_entities)
                            df_emb_entities = self.data.sparkSession.createDataFrame([], schema_df_embeddings)
                            index_row_dataframe = 0

                end = time.time()
                print(end - start)
                if df_emb_entities.count() > 0 or len(ls_emb_aux) > 0:
                    df_emb_entities.union(
                        self.data.sparkSession.createDataFrame(ls_emb_aux, schema_df_embeddings)).write.mode('append').json(path_embedding_entities)

            with open(SparkData.PATH_EMBEDDINGS + 'done.txt', 'w') as f:
                f.write('All entities and literals embedded!')
        #df_emb_entities = self.data.sparkSession.read.schema(schema_df_embeddings).json(path_embedding_entities)
        #print(df_emb_literals.union(df_emb_entities).count())


        print('All entities and literals embedded!')

    def get_embedding(self, sentences):

        response = self.embedding_model.embeddings.create(
            model="text-embedding-3-small",
            input=sentences,
            dimensions=512
        )

        return [row.embedding for row in response.data]

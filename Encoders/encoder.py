import bz2
import os
import re
from concurrent.futures import ProcessPoolExecutor
import math
import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import re
import requests
from joblib import Parallel, delayed
import pickle

class Encoder():

    def __init__(self, kg):
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.embedding_model = encoder
        self.kg = kg

    # def fit(self, entities):
    #
    #
    #     queries = [
    #         f"""
    #         SELECT DISTINCT * WHERE
    #         {{
    #             <{entity}> ?p ?o .
    #             FILTER(!isLiteral(?o)  || (isLiteral(?o) && langMatches(lang(?o), "EN")))
    #         }}""" for entity in entities
    #     ]
    #
    #     responses = []
    #     # for query in queries:
    #     #     r = self.kg.get_from_kg(query)
    #     #     responses.append(r['results']['bindings'])
    #
    #     print('Querying the database')
    #     batch_size = 1000
    #     batch = 0
    #     for i in range(0, len(queries), batch_size):
    #         print("Retrieving data from entities: Lot {} of {}. Entities {} of {}.".format(batch, math.ceil(len(queries)/batch_size), i, len(queries)))
    #
    #         with ProcessPoolExecutor(max_workers=10) as executor:
    #             for r in executor.map(self.kg.get_from_kg, queries[i:i + batch_size]):
    #                 responses.append(r['results']['bindings'])
    #
    #
    #
    #     # with ProcessPoolExecutor(max_workers=5) as executor:
    #     #     for r in executor.map(self.kg.get_from_kg, queries):
    #     #         responses.append(r['results']['bindings'])
    #     #responses = [response['results']['bindings'] for response in responses]
    #
    #     # Encoding literals
    #     print('Encoding Literals')
    #     embeddings = {}
    #     relations = {}
    #     for i in range(len(responses)):
    #         # responses_entity = responses[i]
    #         for response in responses[i]:
    #             value = response['o']['value']
    #             if len(value) < 2 or 'http://' in value: continue
    #             value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()
    #
    #             embeddings[value] = self.fit_transform(tf.constant([value]))
    #
    #             if value in relations:
    #                 relations[value].add(entities[i])
    #             else:
    #                 relations[value] = {entities[i]}
    #
    #     # Encoding Nodes
    #     print('Encoding Nodes')
    #     # queries = [
    #     #     f"""
    #     #     SELECT DISTINCT * WHERE
    #     #     {{
    #     #         <{entity}> ?p ?o .
    #     #         FILTER(!isLiteral(?o)  || (isLiteral(?o) && langMatches(lang(?o), "EN")))
    #     #     }}""" for entity in entities
    #     # ]
    #     #
    #     # responses = []
    #     # with ProcessPoolExecutor(max_workers=10) as executor:
    #     #     for r in executor.map(self.kg.get_from_kg, queries):
    #     #         responses.append(r['results']['bindings'])
    #     # #responses = [response['results']['bindings'] for response in responses]
    #
    #     changed = True
    #     count = 0
    #     while changed and count < 3:
    #         count += 1
    #         changed = False
    #         for i in range(len(responses)):
    #             #responses_entity = responses[i]
    #             related_embeddings = []
    #
    #             for response in responses[i]:
    #                 value = response['o']['value']
    #
    #                 # the first iteration
    #                 if count == 1:
    #                     if len(value) < 2 or 'http://' in value: continue
    #                     value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()
    #
    #                 # the other iterations
    #                 elif 'http://' not in value:
    #                     continue
    #
    #                 if value in embeddings:
    #                     related_embeddings.append(embeddings[value])
    #
    #             if len(related_embeddings) > 0:
    #
    #                 # appends the embedding of entity to calculate the new representation
    #                 if count > 1 and entities[i] in embeddings:
    #                     related_embeddings.append(embeddings[entities[i]])
    #
    #                 mean = tf.reduce_mean(tf.stack(related_embeddings, axis=1), 1)
    #                 if entities[i] not in embeddings or not tf.math.reduce_all(
    #                         tf.equal(embeddings[entities[i]], mean)).numpy():
    #                     changed = True
    #                     embeddings[entities[i]] = tf.reduce_mean(tf.stack(related_embeddings, axis=1), 1)
    #
    #     return embeddings, relations

    def fit(self): #, entities):

        directory = './Temp/'
        files = [f for f in os.listdir(directory) if f.endswith('.bz2')]
        path_entities_literals_file = directory + "entities_literals_file.pkl"
        path_literals_entities_file = directory + "literals_entities_file.pkl"

        entities_literals = {}
        literals_entities = {}

        if not os.path.isfile(path_entities_literals_file) or  not os.path.isfile(path_literals_entities_file):
            for i in range(len(files)):
                source_file = bz2.BZ2File('./Temp/' + files[i], "r")

                source_file.readline()  # It ignores the first line
                next_line = source_file.readline()
                while True:
                    line = next_line
                    next_line = source_file.readline()
                    if len(next_line) == 0:  # This check if the current line is the last and ignore
                        break

                    try:
                        txt = line.decode("utf-8")
                    except:
                        txt = line.decode("iso_8859_1")
                    originalLine = txt

                    #if '@' in txt:
                    if not '@en' in txt:
                        continue

                    txt = re.sub('\r?> <|> "', '|||', txt)
                    txt = re.sub('\r?<', '', txt)
                    txt = re.sub('\r?> .', '', txt)
                    txt = re.sub('\r?\n', '', txt)
                    parts_of_triple = txt.split('|||')

                    if parts_of_triple[0] in entities_literals:
                        entities_literals[parts_of_triple[0]].update({parts_of_triple[2][:-6]})
                    else:
                        entities_literals[parts_of_triple[0]] = {parts_of_triple[2][:-6]}

                    if parts_of_triple[2][:-6] in literals_entities:
                        literals_entities[parts_of_triple[2][:-6]].update({parts_of_triple[0]})
                    else:
                        literals_entities[parts_of_triple[2][:-6]] = {parts_of_triple[0]}

            entities_literals_file = open(path_entities_literals_file, 'wb')
            pickle.dump(entities_literals, entities_literals_file)
            entities_literals_file.close()

            literals_entities_file = open(path_literals_entities_file, 'wb')
            pickle.dump(literals_entities, literals_entities_file)
            literals_entities_file.close()

        else:
            entities_literals_file = open(path_entities_literals_file, 'wb')
            entities_literals = pickle.load(entities_literals_file)
            entities_literals_file.close()

            literals_entities_file = open(path_literals_entities_file, 'wb')
            literals_entities = pickle.load(literals_entities_file)
            literals_entities_file.close()

        return
        # queries = [
        #     f"""
        #      SELECT DISTINCT * WHERE
        #      {{
        #          <{entity}> ?p ?o .
        #          FILTER(!isLiteral(?o)  || (isLiteral(?o) && langMatches(lang(?o), "EN")))
        #      }}""" for entity in entities
        # ]
        #
        responses = []
        #
        # print('Querying the database')
        # batch_size = 1000
        # batch = 0
        # for i in range(0, len(queries), batch_size):
        #     print("Retrieving data from entities: Lot {} of {}. Entities {} of {}.".format(batch, math.ceil(
        #         len(queries) / batch_size), i, len(queries)))
        #
        #     with ProcessPoolExecutor(max_workers=10) as executor:
        #         for r in executor.map(self.kg.get_from_kg, queries[i:i + batch_size]):
        #             responses.append(r['results']['bindings'])

        # Encoding literals
        print('Encoding Literals')
        embeddings = {}
        relations = {}
        for i in range(len(responses)):
            # responses_entity = responses[i]
            for response in responses[i]:
                value = response['o']['value']
                if len(value) < 2 or 'http://' in value: continue
                value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()

                embeddings[value] = self.fit_transform(tf.constant([value]))

                if value in relations:
                    relations[value].add(entities[i])
                else:
                    relations[value] = {entities[i]}

        # Encoding Nodes
        print('Encoding Nodes')

        changed = True
        count = 0
        while changed and count < 3:
            count += 1
            changed = False
            for i in range(len(responses)):
                # responses_entity = responses[i]
                related_embeddings = []

                for response in responses[i]:
                    value = response['o']['value']

                    # the first iteration
                    if count == 1:
                        if len(value) < 2 or 'http://' in value: continue
                        value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()

                    # the other iterations
                    elif 'http://' not in value:
                        continue

                    if value in embeddings:
                        related_embeddings.append(embeddings[value])

                if len(related_embeddings) > 0:

                    # appends the embedding of entity to calculate the new representation
                    if count > 1 and entities[i] in embeddings:
                        related_embeddings.append(embeddings[entities[i]])

                    mean = tf.reduce_mean(tf.stack(related_embeddings, axis=1), 1)
                    if entities[i] not in embeddings or not tf.math.reduce_all(
                            tf.equal(embeddings[entities[i]], mean)).numpy():
                        changed = True
                        embeddings[entities[i]] = tf.reduce_mean(tf.stack(related_embeddings, axis=1), 1)

        return embeddings, relations

    def fit_transform(self, sentences):
        return self.embedding_model(tf.constant(sentences))

    def get_file(remote_file, path_dir):  # , entities, relations):
        try:
            f = open(path_dir + remote_file.split('/')[-1] + ".ttl", "x", encoding="utf-8")
        except:
            print("File " + remote_file.split('/')[-1] + " parsed.")
            return

        if '#' in remote_file:
            return
        print("Downloading file " + remote_file)
        local_file = path_dir + remote_file.split('/')[-1]
        if os.path.isfile(local_file):
            print('File exists')
        else:
            data = requests.get(remote_file)
            with open(local_file, 'wb')as file:
                try:
                    file.write(data.content)
                except:
                    print("fail")

        f.close()

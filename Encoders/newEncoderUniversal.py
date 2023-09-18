import re
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
import tensorflow_hub as hub


class Encoder():

    def __init__(self, kg):
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.embedding_model = encoder
        self.kg = kg

    def fit(self, entities):
        # Encode the literals
        print('Encoding Literals')
        queries = [
            f"""
            SELECT DISTINCT * WHERE 
            {{ 
                <{entity}> ?p ?o . 
                FILTER(isLiteral(?o) && langMatches(lang(?o), "EN"))    
            }}""" for entity in entities
        ]


        responses = []
        with ProcessPoolExecutor(max_workers=10) as executor:
            for r in executor.map(self.kg.get_from_kg, queries):
                responses.append(r)
        responses_entities = [response['results']['bindings'] for response in responses]

        literals_embeddings = {}
        literals_relations = {}
        for i in range(len(responses_entities)):
            responses_entity = responses_entities[i]
            for response_entity in responses_entity:
                value = response_entity['o']['value']
                if len(value) < 2 or 'http://' in value: continue
                value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()

                literals_embeddings[value] = self.fit_transform(tf.constant([value]))

                if value in literals_relations:
                    literals_relations[value].append(entities[i])
                else:
                    literals_relations[value] = [entities[i]]

        # Encode Nodes
        print('Encoding Nodes')
        queries = [
            f"""
                    SELECT DISTINCT * WHERE 
                    {{ 
                        <{entity}> ?p ?o .     
                    }}""" for entity in entities
        ]

        responses = []
        with ProcessPoolExecutor(max_workers=10) as executor:
            for r in executor.map(self.kg.get_from_kg, queries):
                responses.append(r)
        responses_entities = [response['results']['bindings'] for response in responses]

        changed = True
        count = 0
        while changed and count < 3:
            count += 1
            changed = False
            for i in range(len(responses_entities)):
                responses_entity = responses_entities[i]
                embeddings = []

                for response_entity in responses_entity:
                    value = response_entity['o']['value']
                    if len(value) < 2 or 'http://' in value: continue
                    value = re.sub(r'[^a-zA-Z0-9 ]+', '', value).strip()

                    if value in literals_embeddings:
                        embeddings.append(literals_embeddings[value])

                if len(embeddings) > 0:
                    if entities[i] not in literals_embeddings or not tf.math.reduce_all(
                            tf.equal(literals_embeddings[entities[i]],
                                     tf.reduce_mean(tf.stack(embeddings, axis=1), 1))).numpy():
                        changed = True
                        literals_embeddings[entities[i]] = tf.reduce_mean(tf.stack(embeddings, axis=1), 1)

        return literals_embeddings, literals_relations

    def fit_transform(self, sentences):
        return self.embedding_model(tf.constant(sentences))

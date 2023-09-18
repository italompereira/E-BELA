import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class Encoder():

    def __init__(self):
        encoder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.embedding_model = encoder


    def fit(self, kg, entities):

        # Encode the literals
        print('Enconding Literals')
        queries = [
            f"""
            SELECT * WHERE 
            {{ 
                <{entity}> ?p ?o . 
                FILTER(isLiteral(?o) && langMatches(lang(?o), "EN"))    
            }}""" for entity in entities
        ]
        responses = [kg.connector.fetch(query) for query in queries]
        responses_entities = [response['results']['bindings'] for response in responses]

        literals_embeddings = {}
        literals_relations = {}
        count = 0
        for i in range(len(responses_entities)):
            #literals[entities[i]] = []
            responses_entity = responses_entities[i]
            sentences = []
            for response_entity in responses_entity:
                sentences.append(response_entity['o']['value'])
            if len(sentences) == 0:
                print(count)
                print(entities[i])
                count += 1
                continue
            literals_embeddings[entities[i]] = self.fit_transform(tf.constant(sentences))
            literals_relations[entities[i]] = [].append(entities[i])

        return literals_embeddings

    def fit_transform(self, sentences):
        return self.embedding_model(tf.constant(sentences))
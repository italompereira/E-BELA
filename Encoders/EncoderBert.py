import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

class Encoder():

    def __init__(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
        preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        encoder_inputs = preprocessor(text_input)
        encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
            trainable=True)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].

        self.embedding_model = tf.keras.Model(text_input, pooled_output)


    def fit(self, kg, entities):


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

        literals = {}
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
            literals[entities[i]] = self.fit_transform(tf.constant(sentences))

        return literals

    def fit_transform(self, sentences):
        return self.embedding_model(tf.constant(sentences))
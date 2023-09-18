import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from Encoders.newEncoderUniversal import Encoder
from kg import KG
import logging

logging.basicConfig(level=logging.INFO, filename="program.log", format="%(asctime)s - %(levelname)s - %(message)s")

class NewRDF2VEC():
    def __init__(self, endpoint, embeddings_file, literal_relations_file):
        self.embeddings_file = embeddings_file
        self.literal_relations_file = literal_relations_file
        self.kg = KG(endpoint, is_remote=True, skip_verify=True)
        self.entities, _, _ = Data(self.kg).load_entities()

    def __call__(self, *args, **kwargs):
        print('Generating embedding ... ')
        transformer = Encoder(self.kg)
        literals_embeddings, literals_relations = transformer.fit(self.entities)

        try:
            with open(embeddings_file, 'w', encoding='utf8')as file: # saves the embeddings of literals and URIs
                for key in literals_embeddings:
                    for embedding in literals_embeddings[key]:
                        file.write((key + '|||' + ' '.join(map(str, embedding.numpy()))) + '\n')
            with open(literal_relations_file, 'w', encoding='utf8')as file: # saves the related URIs from literals
                for key in literals_relations:
                    file.write((key + '|||' + '|||'.join(map(str, literals_relations[key]))) + '\n')
        except Exception as e:
            print(e)
            logging.info(f"Message: {e}")


class Data:
    def __init__(self, kg, check_on_endpoint=True):
        self.check_on_endpoint = check_on_endpoint
        self.dataset = self.load_dataset()
        self.kg = kg

    def get_triples(self, sentence):
        sentence = re.sub('\r?\n', ' ', sentence)
        sentence = re.findall('\{.*?\}', sentence)[0][1:-1].rstrip(". ").strip()  # Get substring inside { and }
        sentence = re.sub('\r? \. ', '|||', sentence)  # Replace space_dot_space for |||
        sentence = re.sub('\r?\. <', '|||<', sentence)  # Replace dot_< for |||
        sentence = re.sub('\r?\. \?', '|||?', sentence)  # Replace dot_? for |||
        sentence = re.sub('\r?>\. ', '|||?', sentence)  # Replace dot_? for |||
        sentence = re.sub('\r?>|<', '', sentence)  # Remove < OR >
        sentence = sentence.split('|||')
        return sentence

    def load_dataset(self):
        path_dataset = './LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json'

        with open(path_dataset, encoding='utf8') as data_file:
            dataset = json.load(data_file)

        return dataset

    def load_grounth_truth(self):
        entity_mapping = {}
        predicate_mapping = {}
        for text in self.dataset:
            for mapping in text['entity mapping']:
                entity_mapping[mapping['label']] = mapping['uri']

        for text in self.dataset:
            for mapping in text['predicate mapping']:
                if 'label' in mapping:
                    predicate_mapping[mapping['label']] = mapping['uri']

        return entity_mapping, predicate_mapping

    def load_entities(self):

        entities = []
        relations = []

        all_triples = [self.get_triples(text['sparql_query']) for text in self.dataset]
        for triples in all_triples:
            # print('--')
            for triple in triples:
                # print(triple)
                if not 'filter' in triple.lower():
                    rdf_triple = triple.split(' ')

                if 'http://' in rdf_triple[0]:
                    if 'resource' in rdf_triple[0]:
                        entities.append(rdf_triple[0])
                    else:
                        relations.append(rdf_triple[0])

                if 'http://' in rdf_triple[2]:
                    if 'resource' in rdf_triple[2]:
                        entities.append(rdf_triple[2])
                    else:
                        relations.append(rdf_triple[2])

                if 'http://' in rdf_triple[1]:
                    relations.append(rdf_triple[1])

        entities = list(set(entities))
        relations = list(set(relations))

        all_entities_relations = sorted(list(set(entities + relations)))

        if self.check_on_endpoint:
            print('Checking entities in Virtuoso ...')
            try:
                print('Opening entities_not_in_virtuoso.txt.')
                with open("entities_not_in_virtuoso.txt", 'r', encoding='utf8') as file:
                    print('Reading lines from entities_not_in_virtuoso.txt.')
                    entities_not_in_list = file.readlines()
                    index_entities_not_in_list = [all_entities_relations.index(e[:-1]) for e in entities_not_in_list]
            except Exception as e:
                print(e)
                print('File entities_not_in_virtuoso.txt does not exists.')
                entities_not_in_list, index_entities_not_in_list = self.is_exist(all_entities_relations)
                print('Creating file entities_not_in_virtuoso.txt.')
                try:
                    with open("entities_not_in_virtuoso.txt", 'w', encoding='utf8')as file:
                        for element in entities_not_in_list:
                            file.write(element + '\n')
                except Exception as ex:
                    print(ex)
                    logging.info(f"Message: {ex}")
                logging.info(f"Message: {e}")

            for i in sorted(index_entities_not_in_list, reverse=True):
                del all_entities_relations[i]
        return all_entities_relations, entities, relations

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

class CheckEmbeddings:
    ONLY_URIS = 0
    ONLY_LITERALS = 1
    ALL = 2

    def __init__(self, endpoint, embeddings_file, literal_relations_file, embeddings_of):
        self.embeddings_list = self.get_embeddings(embeddings_file, embeddings_of)
        if embeddings_of != CheckEmbeddings.ONLY_URIS:
            self.literal_relations = self.get_literals_relations(literal_relations_file)

        self.kg = KG(endpoint, is_remote=True, skip_verify=True)
        data = Data(self.kg)
        self.entities, _, _ = data.load_entities()
        self.entities_mapping, self.predicates_mapping = data.load_grounth_truth()


    def __call__(self, *args, **kwargs):
        print('Evaluating embeddings.')
        self.evaluate()
        # uri_entities = {}
        #
        # if not os.path.isfile('uri_entities.sav'):
        #
        #     sample_size = 5
        #
        #     type_entities = {
        #         "http://dbpedia.org/ontology/Country": "red",
        #         "http://dbpedia.org/ontology/City": "yellow",
        #         "http://dbpedia.org/ontology/Person": "gray",
        #         "http://dbpedia.org/ontology/Film": "purple"
        #     }
        #
        #     for uri, color in type_entities.items():
        #         limit = 10000
        #         offset = 0
        #         count = 0
        #
        #         while (True):
        #
        #             query = f"""SELECT * WHERE
        #                 {{
        #                     {{
        #                         ?s a <{uri}> .
        #                         ?s <http://www.w3.org/2000/01/rdf-schema#label> ?o .
        #                     }}
        #
        #                 }} offset {offset} limit {limit}"""
        #
        #             offset += limit
        #
        #             responses = self.kg.connector.fetch(query)
        #             if len(responses['results']['bindings']) == 0 or count >= sample_size:
        #                 break
        #
        #             # sample_of_population = (len(responses) <  sample_size) ? responses : responses['results']['bindings'], sample_size
        #
        #             for el in responses['results']['bindings']:
        #                 # uri_entities.append(el['s']['value'])
        #                 # labels.append(el['o']['value'])
        #                 # colors.append(color)
        #
        #                 if el['s']['value'] not in self.entities:
        #                     continue
        #
        #                 count += 1
        #                 uri_entities[el['s']['value']] = {}
        #                 uri_entities[el['s']['value']]['label'] = el['o']['value']
        #                 uri_entities[el['s']['value']]['color'] = color
        #
        #                 if count >= sample_size:
        #                     break
        # else:
        #     uri_entities = joblib.load('uri_entities.sav')

        # check_top_k_between_entities(list(uri_entities))

        # self.check_top_k_by_sentences(
        #     ['clay Aiken','relatives','Tukwila, Washington',"US","Martin Ferguson","Labour Party","Labour Party (Norway)","Ringgold High School"]
        #     # ,'Congressional Black Caucus', 'Delta Air Lines', 'moutpiece', 'periodical literature', 'united states', 'usa',
        #     # 'american actor', 'barack', 'obama', 'back to the future', 'barack obama', 'donald trump', 'michele obama',
        #     # 'president usa']
        # )

        # plot(uri_entities)

        # print(responses)

    def evaluate(self, ):
        transformer = Encoder(self.kg)
        print("Embedding " + str(len(self.entities_mapping)) + " entities from Ground Truth.")
        vectors = transformer.fit_transform(list(self.entities_mapping))

        y_true = [i for i in self.entities_mapping.values()]

        y_pred = []
        for i in range(len(vectors)):
            if i % 10 == 0:
                print(str(i) + " vectors processed of " + str(len(vectors)), end=".\t")
            y_pred.append(self.get_similars_to(vectors[i], 10))
            if i % 100 == 0:
                print("\n")

        print('Accuracy: ')
        print(sum(1 for x,y in zip(y_true,y_pred) if x == y) / len(y_true))

    def get_similars_to(self, vector, k):

        embedded_entities = [i[0] for i in self.embeddings_list]

        distance_array = np.zeros(len(embedded_entities))
        for i in range(len(self.embeddings_list)):
            distance_array[i] = self.calc_distance((i, vector))

        # The top_k function retrieves the largest values and in this case we need the smallest ones, for this the sign
        # of the distance measurements is changed to negative to retrieve those closest to zero
        top_k = tf.math.top_k(tf.negative(distance_array), k=k)
        indexes = top_k[1].numpy()
        ranking_list_aux = [embedded_entities[i] for i in indexes]

        # ranking_list = []
        # for i in range(len(ranking_list_aux)):
        #     res_entity = ranking_list_aux[i]
        #     if not (res_entity.startswith("https:") or res_entity.startswith("http:")) and res_entity in self.literal_relations:
        #         for entity in self.literal_relations[res_entity]:
        #             ranking_list.append(entity)
        #     else:
        #         ranking_list.append(res_entity)
        # print(ranking_list[0])
        #
        # # Sort by weights
        # ranking_list = {}
        # weight = k
        # for i in range(len(ranking_list_aux)):
        #     res_entity = ranking_list_aux[i]
        #     if not (res_entity.startswith("https:") or res_entity.startswith("http:")) and res_entity in self.literal_relations:
        #         for entity in self.literal_relations[res_entity]:
        #             ranking_list[entity] = weight if entity not in ranking_list else ranking_list[entity] + weight
        #     else:
        #         ranking_list[res_entity] = weight if res_entity not in ranking_list else ranking_list[res_entity] + weight
        #     weight -= 1
        # print(max(ranking_list, key=ranking_list.get))

        ranking_list = {}
        for i in range(len(ranking_list_aux)):
            res_entity = ranking_list_aux[i]
            if not (res_entity.startswith("https:") or res_entity.startswith("http:")) and res_entity in self.literal_relations:
                for entity in self.literal_relations[res_entity]:
                    ranking_list[entity] = 1 / (i+1) if entity not in ranking_list else ranking_list[entity] + ranking_list[entity] / (i+1)
            else:
                ranking_list[res_entity] = 1 / (i+1) if res_entity not in ranking_list else ranking_list[res_entity] + ranking_list[res_entity] / (i+1)
        # print(max(ranking_list, key=ranking_list.get))

        return max(ranking_list, key=ranking_list.get)


    def calc_distance(self, arg):
        # Similaridade do cosseno
        vector_b = self.embeddings_list[arg[0]][1]
        vector_a = arg[1]
        return tf.keras.losses.CosineSimilarity(axis=0)(vector_a, vector_b).numpy()

        # Distância Euclidiana
        # return tf.norm(vector_a - vector_b, ord='euclidean')

    def check_top_k_by_sentences(self, sentences):
        transformer = Encoder(self.kg)
        vectors = transformer.fit_transform(sentences)

        # vector = tf.constant(pre_trained_model['obama'])
        # get_most_similar_to(vector, 50, embeddings_file)
        #
        # vector = tf.constant(pre_trained_model['barack'])
        for i in range(len(vectors)):
            print(sentences[i])
            self.get_most_similar_to(vectors[i], 10)
            print('-----\n')

    # def check_top_k_between_entities(self, uri_entities) -> None:
    #
    #     entities = [i[0] for i in self.embeddings_list]
    #     distance_matrix = []
    #
    #     id_entities = sum(
    #         [[idx for idx, value in enumerate(entities) if value == uri_entity] for uri_entity in uri_entities],
    #         [])  # [entities.index(uri_entity) for uri_entity in uri_entities if uri_entity in entities]
    #
    #     entities_list = [self.embeddings_list[i] for i in id_entities]
    #
    #     for i in range(len(entities_list)):
    #         distance_matrix.append([])
    #         # if i not in id_entities:
    #         #     continue
    #         for j in range(len(entities_list)):
    #             distance_matrix[i].append(self.calc_distance(entities_list[i][1], entities_list[j][1]))
    #
    #     # top_k = []
    #     for i in range(len(entities_list)):
    #         # top_k = tf.math.top_k(tf.negative(distance_matrix[i]), k=15)
    #         top_k = tf.math.top_k(distance_matrix[i], k=15)
    #         #
    #         indexes = top_k[1].numpy()
    #
    #         queries = [
    #             f"""SELECT * WHERE
    #             {{
    #                 {{
    #                     <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?o .
    #                 }}
    #                 union
    #                 {{
    #                     <{entity}> <http://dbpedia.org/ontology/description> ?o .
    #                 }}
    #                 union
    #                 {{
    #                     <{entity}> <http://dbpedia.org/ontology/alias> ?o .
    #                 }}
    #                 union
    #                 {{
    #                     <{entity}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o .
    #                 }}
    #             }}
    #             """ for entity in [entities_list[i][0] for i in indexes]
    #         ]
    #
    #         print(entities_list[i][0])
    #         responses = [self.kg.connector.fetch(query) for query in queries]
    #
    #         for response in responses:
    #             for el in response['results']['bindings']:
    #                 print(el['o']['value'], end=" -- ")
    #             print(' ')
    #         print(' ')

    # def plot(self, uri_entities):
    #
    #     entities = [i[0] for i in self.embeddings_list]
    #
    #     id_entities = sum(
    #         [[idx for idx, value in enumerate(entities) if value == uri_entity] for uri_entity in uri_entities], [])
    #     entities_list = [self.embeddings_list[i] for i in id_entities]
    #     embeddings = []
    #     colors = []
    #     labels = []
    #
    #     for i in range(len(entities_list)):
    #         embeddings.append(entities_list[i][1])
    #         colors.append(uri_entities[entities_list[i][0]]['color'])
    #         labels.append(uri_entities[entities_list[i][0]]['label'])
    #
    #     walk_tsne = TSNE(random_state=5)
    #     X_tsne = walk_tsne.fit_transform(np.array(embeddings))
    #
    #     plt.figure(figsize=(15, 15))
    #     plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)
    #
    #     for i, txt in enumerate(labels):
    #         plt.annotate(txt, (X_tsne[:, 0][i], X_tsne[:, 1][i]))
    #
    #     plt.show()

    def get_embeddings(self, embeddings_file, embeddings_of):
        embeddings_list = []

        with open(embeddings_file, encoding='utf8') as embedding_file:
            lines = embedding_file.readlines()

            for i in range(len(lines)):

                if embeddings_of == CheckEmbeddings.ONLY_URIS:
                    if not (lines[i].startswith("https:") or lines[i].startswith("http:")):
                        continue
                elif embeddings_of == CheckEmbeddings.ONLY_LITERALS:
                    if (lines[i].startswith("https:") or lines[i].startswith("http:")):
                        continue

                line_splitted = lines[i].split('|||')
                try:
                    embeddings_list.append((line_splitted[0], tf.constant([float(x) for x in line_splitted[1].split(' ')])))
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

    def get_most_similar_to(self, vector, k):
        # entities_list = [i for i in entities_list if  (i[0].startswith("https:") or i[0].startswith("http:"))]

        entities = [i[0] for i in self.embeddings_list]

        distance_array = np.zeros(len(entities))
        for i in range(len(self.embeddings_list)):
            distance_array[i] = self.calc_distance((i, vector))
        #
        # teste = []
        # with ProcessPoolExecutor(max_workers=10) as executor:
        #     for i in executor.map(self.teste, range(100)):
        #         print(i)
        #         teste.append(i)

        # with ProcessPoolExecutor(max_workers=2) as executor:
        #     for i, r in zip(range(10), executor.map( self.teste, range(10), repeat('a'))):
        #         print(i, r)


        # distance_array = np.zeros(len(entities))
        # with ProcessPoolExecutor(max_workers=10) as executor:
        #     for i, r in zip(range(len(self.entities_list)), executor.map( self.calc_distance, range(len(self.entities_list)), 10 )):
        #         distance_array[i] = r


        # The top_k function retrieves the largest values and in this case we need the smallest ones, for this the sign of
        # the distance measurements is changed to negative to retrieve those closest to zero
        top_k = tf.math.top_k(tf.negative(distance_array), k=k)

        # Get the largest values of similarity, close to one
        # top_k = tf.math.top_k(distance_vector, k=k)

        indexes = top_k[1].numpy()

        # queries = [
        #     f"""SELECT * WHERE
        #             {{
        #                 {{
        #                     <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?o .
        #                 }}
        #                 union
        #                 {{
        #                     <{entity}> <http://dbpedia.org/ontology/description> ?o .
        #                 }}
        #                 union
        #                 {{
        #                     <{entity}> <http://dbpedia.org/ontology/alias> ?o .
        #                 }}
        #                 union
        #                 {{
        #                     <{entity}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o .
        #                 }}
        #             }}
        #             """ for entity in [entities[i] for i in indexes]
        # ]

        for i in indexes:

            entity = entities[i]
            print(entity, end=" -- ")
            if entity.startswith("https:") or entity.startswith("http:"):

                query = f"""SELECT * WHERE 
                                {{ 
                                    {{
                                        <{entity}> <http://www.w3.org/2000/01/rdf-schema#label> ?o .
                                    }}
                                    union
                                    {{                     
                                        <{entity}> <http://dbpedia.org/ontology/description> ?o .
                                    }}
                                    union
                                    {{                     
                                        <{entity}> <http://dbpedia.org/ontology/alias> ?o .
                                    }}
                                    union
                                    {{
                                        <{entity}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?o .
                                    }}
                                }}
                                """

                try:
                    response = self.kg.connector.fetch(query)

                    for el in response['results']['bindings']:
                        print(el['o']['value'], end=" -- ")
                    print(' ')

                except Exception as e:
                    print(entity)
                    logging.info(f"Message: {e}")
                    # print(' ')
            else:
                print()

        # responses = [kg.connector.fetch(query) for query in queries]
        #
        # for response in responses:
        #     for el in response['results']['bindings']:
        #         print(el['o']['value'], end=" -- ")
        #     print(' ')
        # print(' ')
        return


if __name__ == "__main__":
    embeddings_file = 'Embeddings/new_embedding_vectors_universal_3_dbpedia.txt'
    literal_relations_file = 'Embeddings/literal_relations_dbpedia.txt'
    endpoint = "https://dbpedia.org/sparql"
    if not os.path.isfile(embeddings_file):
        newRDF2VEC = NewRDF2VEC(endpoint, embeddings_file, literal_relations_file)
        newRDF2VEC()

    check = CheckEmbeddings(endpoint, embeddings_file, literal_relations_file, CheckEmbeddings.ALL)
    check()

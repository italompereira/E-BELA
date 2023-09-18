import json
import os
import re
import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

from Encoders.EncoderBert import Encoder
from kg import KG


def calc_distance(vector_a, vector_b):
    return tf.norm(vector_a - vector_b, ord='euclidean')


def check_quality_of_embeddings(embeddings_file):
    entities, relations, all = load_entities()
    uri_entities = {}

    if not os.path.isfile('uri_entities.sav'):

        sample_size = 5
        kg = load_model()

        type_entities = {
            "http://dbpedia.org/ontology/Country": "red",
            "http://dbpedia.org/ontology/City": "yellow",
            "http://dbpedia.org/ontology/Person": "gray",
            "http://dbpedia.org/ontology/Film": "purple"
        }

        for uri, color in type_entities.items():
            limit = 10000
            offset = 0
            count = 0

            while (True):

                query = f"""SELECT * WHERE 
                    {{
                        {{
                            ?s a <{uri}> .
                            ?s <http://www.w3.org/2000/01/rdf-schema#label> ?o .
                        }}

                    }} offset {offset} limit {limit}"""

                offset += limit

                responses = kg.connector.fetch(query)
                if len(responses['results']['bindings']) == 0 or count >= sample_size:
                    break

                # sample_of_population = (len(responses) <  sample_size) ? responses : responses['results']['bindings'], sample_size

                for el in responses['results']['bindings']:
                    # uri_entities.append(el['s']['value'])
                    # labels.append(el['o']['value'])
                    # colors.append(color)

                    if el['s']['value'] not in entities:
                        continue

                    count += 1
                    uri_entities[el['s']['value']] = {}
                    uri_entities[el['s']['value']]['label'] = el['o']['value']
                    uri_entities[el['s']['value']]['color'] = color

                    if count >= sample_size:
                        break
    else:
        uri_entities = joblib.load('uri_entities.sav')

    # check_top_k_between_entities(list(uri_entities), embeddings_file)
    check_top_k_by_sentences(
        ['Congressional Black Caucus', 'Delta Air Lines', 'moutpiece', 'periodical literature', 'united states', 'usa',
         'american actor', 'barack', 'obama', 'back to the future', 'barack obama', 'donald trump', 'michele obama',
         'president usa'])
    plot(uri_entities)


def check_top_k_by_sentences(sentences):
    transformer = Encoder()
    vectors = transformer.fit_transform(sentences)

    for vector in vectors:
        get_most_similar_to(vector, 50, embeddings_file)


def check_top_k_between_entities(uri_entities, embeddings_file) -> None:
    entities_list = get_embeddings(embeddings_file)

    entities = [i[0] for i in entities_list]
    distance_matrix = []

    id_entities = sum(
        [[idx for idx, value in enumerate(entities) if value == uri_entity] for uri_entity in uri_entities],
        [])  # [entities.index(uri_entity) for uri_entity in uri_entities if uri_entity in entities]

    entities_list = [entities_list[i] for i in id_entities]

    for i in range(len(entities_list)):
        distance_matrix.append([])
        # if i not in id_entities:
        #     continue
        for j in range(len(entities_list)):
            distance_matrix[i].append(calc_distance(entities_list[i][1], entities_list[j][1]))

    # top_k = []
    for i in range(len(entities_list)):
        top_k = tf.math.top_k(tf.negative(distance_matrix[i]), k=15)
        #
        indexes = top_k[1].numpy()

        queries = [
            f"""SELECT * WHERE 
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
            """ for entity in [entities_list[i][0] for i in indexes]
        ]

        print(entities[id_entities[1]])
        kg = load_model()
        responses = [kg.connector.fetch(query) for query in queries]

        for response in responses:
            for el in response['results']['bindings']:
                print(el['o']['value'], end=" -- ")
            print(' ')
        print(' ')


def generate_embeddings(embeddings_file):
    print('Loading model ...')
    kg = load_model()

    print('Getting entities ...')
    entities, relations, all = load_entities()
    # entities = list(set(entities + relations))[0:10]

    print('Generating embedding ... ')

    from Encoders.EncoderUniversal import Encoder
    transformer = Encoder()

    from Encoders.EncoderBert import Encoder
    transformer = Encoder()

    print('Checking entities in Virtuoso ...')
    try:
        with open("entities_not_in_virtuoso.txt", 'r', encoding='utf8') as file:
            entities_not_in_list = file.readlines()
            index_entities_not_in_list = [entities.index(e[:-1]) for e in entities_not_in_list]
    except Exception as e:
        entities_not_in_list, index_entities_not_in_list = is_exist(kg, entities)
        print('Elements not in list')
        try:
            with open("entities_not_in_virtuoso.txt", 'w', encoding='utf8')as file:
                for element in entities_not_in_list:
                    file.write(element + '\n')
        except Exception as e:
            print(e)

    for i in sorted(index_entities_not_in_list, reverse=True):
        del entities[i]
    embeddings = transformer.fit(kg, entities)

    try:
        with open(embeddings_file, 'w', encoding='utf8')as file:
            for key in embeddings:
                for embedding in embeddings[key]:
                    file.write((key + ' ' + ' '.join(map(str, embedding.numpy()))) + '\n')
        # print(entities[i], *embeddings[i])
    except Exception as e:
        print(e)


def get_embeddings(embeddings_file):
    entities_list = []

    with open(embeddings_file) as embedding_file:
        lines = embedding_file.readlines()
        # distance_matrix = np.zeros((len(lines), len(lines)))

        for i in range(len(lines)):
            line_splitted = lines[i].split(' ')
            entities_list.append((line_splitted[0], tf.constant([float(x) for x in line_splitted[1:]])))

    return entities_list


def get_most_similar_to(vector, k, embeddings_file):
    entities_list = get_embeddings(embeddings_file)
    entities = [i[0] for i in entities_list]

    distance_vector = np.zeros(len(entities))
    for i in range(len(entities_list)):
        distance_vector[i] = calc_distance(vector, entities_list[i][1])

    top_k = tf.math.top_k(tf.negative(distance_vector), k=k)
    indexes = top_k[1].numpy()

    queries = [
        f"""SELECT * WHERE 
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
                """ for entity in [entities[i] for i in indexes]
    ]

    kg = load_model()
    responses = [kg.connector.fetch(query) for query in queries]

    for response in responses:
        for el in response['results']['bindings']:
            print(el['o']['value'], end=" -- ")
        print(' ')
    print(' ')
    return


def get_triples(sentence):
    sentence = re.sub('\r?\n', ' ', sentence)
    sentence = re.findall('\{.*?\}', sentence)[0][2:-1].rstrip(". ")  # Get substring inside { and }
    sentence = re.sub('\r? \. ', '|||', sentence)  # Replace space_dot_space for |||
    sentence = re.sub('\r?\. <', '|||<', sentence)  # Replace dot_< for |||
    sentence = re.sub('\r?\. \?', '|||?', sentence)  # Replace dot_? for |||
    sentence = re.sub('\r?>\. ', '|||?', sentence)  # Replace dot_? for |||
    sentence = re.sub('\r?>|<', '', sentence)  # Remove < OR >
    sentence = sentence.split('|||')
    return sentence


def load_entities():
    path_train = 'train.json'
    path_test = 'test.json'

    with open(path_train) as data_file:
        train = json.load(data_file)

    with open(path_test) as data_file:
        test = json.load(data_file)

    all_dataset = train + test

    entities = []
    relations = []

    all_triples = [get_triples(text['sparql_dbpedia18']) for text in all_dataset]
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
    return entities, relations, all_dataset


def load_model():
    label_predicates = [
        'http://dbpedia.org/ontology/alias',
        'http://dbpedia.org/ontology/description',
        'http://www.w3.org/2000/01/rdf-schema#label',
        'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
        'http://www.w3.org/2000/01/rdf-schema#subClassOf',
        'http://www.w3.org/2002/07/owl#sameAs',
        # 'http://dbpedia.org/ontology/description',
        # 'http://dbpedia.org/ontology/country',
        # 'http://dbpedia.org/ontology/birthDate',
        # 'http://dbpedia.org/ontology/deathDate',
        # 'http://dbpedia.org/ontology/releaseDate',
        # 'http://dbpedia.org/ontology/formationDate',
        # 'http://dbpedia.org/ontology/date',
        # 'http://dbpedia.org/ontology/endDate',
        # 'http://dbpedia.org/ontology/startDate',
        # 'http://dbpedia.org/ontology/landingDate',
        # 'http://dbpedia.org/ontology/launchDate'
    ]

    # KG Loading Alternative 2: Using a dbpedia endpoint (nothing is loaded into memory)
    kg = KG("http://200.131.10.200:8890/sparql", is_remote=True, skip_verify=True, include_predicates=label_predicates)
    return kg


def is_exist(kg, entities):
    queries = [
        f"ASK WHERE {{ <{entity}> ?p ?o . }}" for entity in entities
    ]

    responses = [kg.connector.fetch(query) for query in queries]
    responses = [res["boolean"] for res in responses]

    index_entities_not_in_list = [i for i, val in enumerate(responses) if not val]
    return [entities[i] for i in index_entities_not_in_list], index_entities_not_in_list


def plot(uri_entities):
    entities_list = get_embeddings(embeddings_file)

    entities = [i[0] for i in entities_list]

    id_entities = sum(
        [[idx for idx, value in enumerate(entities) if value == uri_entity] for uri_entity in uri_entities], [])
    entities_list = [entities_list[i] for i in id_entities]
    embeddings = []
    colors = []
    labels = []

    for i in range(len(entities_list)):
        embeddings.append(entities_list[i][1])
        colors.append(uri_entities[entities_list[i][0]]['color'])
        labels.append(uri_entities[entities_list[i][0]]['label'])

    walk_tsne = TSNE(random_state=5)
    X_tsne = walk_tsne.fit_transform(np.array(embeddings))

    plt.figure(figsize=(15, 15))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors)

    for i, txt in enumerate(labels):
        plt.annotate(txt, (X_tsne[:, 0][i], X_tsne[:, 1][i]))

    plt.show()


if __name__ == "__main__":
    embeddings_file = 'backup/embedding_vectors_bert.txt'
    if not os.path.isfile(embeddings_file):
        generate_embeddings(embeddings_file)
    generate_embeddings(embeddings_file)
    check_quality_of_embeddings(embeddings_file)

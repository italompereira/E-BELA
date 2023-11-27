import os

from evaluate import Evaluate
from rdf2vec_dbpedia import NewRDF2VEC

if __name__ == "__main__":
    embeddings_file = 'Embeddings/new_embedding_vectors_universal_3_dbpedia.txt'
    literal_relations_file = 'Embeddings/literal_relations_dbpedia.txt'
    path_files = 'Embeddings/'
    endpoint = "http://200.131.10.200:8890/sparql" #"https://dbpedia.org/sparql"
    if not os.path.isfile(embeddings_file):
        newRDF2VEC = NewRDF2VEC(endpoint, embeddings_file, literal_relations_file)
        newRDF2VEC()
    newRDF2VEC = NewRDF2VEC(endpoint, embeddings_file, literal_relations_file)
    newRDF2VEC()

    # kg = KG(endpoint, is_remote=True, skip_verify=True)
    # data = Data(kg)
    # entities_mapping, predicates_mapping = data.load_conll_grounth_truth()

    #evaluate = Evaluate(endpoint, embeddings_file, literal_relations_file, path_files, Evaluate.ALL)
    #evaluate()
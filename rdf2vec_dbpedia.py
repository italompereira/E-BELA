import bz2
import json
import logging
import os
import pickle
import re
import time
import requests
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor
from random import sample
from Encoders.encoder import Encoder
from data import Data
from kg import KG

logging.basicConfig(level=logging.INFO, filename="program.log", format="%(asctime)s - %(levelname)s - %(message)s")


class NewRDF2VEC():
    def __init__(self, endpoint, embeddings_file, literal_relations_file):
        self.embeddings_file = embeddings_file
        self.literal_relations_file = literal_relations_file
        self.kg = KG(endpoint, is_remote=True, skip_verify=True)
        #self.entities, _, _ = Data(self.kg).load_entities()
            #"https://downloads.dbpedia.org/2016-10/core/labels_en.ttl.bz2")

    def __call__(self, *args, **kwargs):
        print('Generating embedding ... ')
        transformer = Encoder(self.kg)
        literals_embeddings, literals_relations = transformer.fit() # self.entities)
        return
        try:
            with open(self.embeddings_file, 'w', encoding='utf8')as file:  # saves the embeddings of literals and URIs
                for key in literals_embeddings:
                    for embedding in literals_embeddings[key]:
                        file.write((key + '|||' + ' '.join(map(str, embedding.numpy()))) + '\n')
            with open(self.literal_relations_file, 'w', encoding='utf8')as file:  # saves the related URIs from literals
                for key in literals_relations:
                    file.write((key + '|||' + '|||'.join(map(str, literals_relations[key]))) + '\n')
        except Exception as e:
            print(e)
            logging.info(f"Message: {e}")




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
    def __init__(self, endpoint, data):
        try:
            self.kg = KG(endpoint, is_remote=True, skip_verify=True)
        except Exception as ex:
            self.kg = None
            print(ex)
            logging.info(f"Message: {ex}")
        self.data = data

    #@tf.function(jit_compile=True)
    def __call__(self, *args, **kwargs):
        print('Generating embedding ... ')
        transformer = Encoder(self.kg, self.data)
        try:
            transformer.fit()
        except Exception as ex:
            print(ex)
            logging.info(f"Message: {ex}")
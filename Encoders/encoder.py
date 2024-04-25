import logging

from Dumps.spark_data import SparkData
from Encoders.encoder_sentence_transformer import EncoderTransformer
from Encoders.encoder_universal_sentence_encoder import EncoderUSE
from EndpointConnector.kg import KG

logging.basicConfig(level=logging.INFO, filename="program.log", format="%(asctime)s - %(levelname)s - %(message)s")

class Encoder():
    def __init__(self, spark_data):
        self.spark_data = spark_data

    #@tf.function(jit_compile=True)
    def __call__(self, *args, **kwargs):
        print('Generating embedding ... ')
        #transformer = EncoderUSE(self.spark_data)
        transformer = EncoderTransformer(self.spark_data)
        try:
            transformer.fit('average')
        except Exception as ex:
            print(ex)
            logging.info(f"Message: {ex}")
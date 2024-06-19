import logging

from Dumps.spark_data import SparkData
from Encoders.encoder_sentence_transformer import EncoderTransformer
from Encoders.encoder_universal_sentence_encoder import EncoderUSE
from EndpointConnector.kg import KG

logging.basicConfig(level=logging.INFO, filename="program.log", format="%(asctime)s - %(levelname)s - %(message)s")

class Encoder():
    def __init__(self, spark_data, model: "ST", weighted:False, strategy: "AVG", top_n:None, top_n_way:None, config_experiment: ''):
        self.spark_data = spark_data
        self.model = model
        self.weighted = weighted
        self.strategy = strategy
        self.top_n = top_n
        self.top_n_way = top_n_way
        self.config_experiment = config_experiment
        self.transformer = EncoderTransformer(self.spark_data, self.model, self.weighted, self.strategy, self.top_n, self.top_n_way, self.config_experiment)

    #@tf.function(jit_compile=True)
    def __call__(self, *args, **kwargs):
        print('Generating embedding ... ')

        try:
            self.transformer.fit()
        except Exception as ex:
            print(ex)
            logging.info(f"Message: {ex}")
            raise ex
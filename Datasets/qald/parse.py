import json
import regex as re

class Parse:

    PATHS = {
        #'qald-7': 'qald-7-test-multilingual.json',
        'qald-8': 'qald-7-test-multilingual.json',
        'qald-9': 'qald-7-test-multilingual.json',
    }

    def __init__(self):
        self.dataset = self.parse_dataset()

    def parse_dataset(self):
        #path_dataset = './Datasets/LC-QuADAnnotated/FullyAnnotated_LCQuAD5000.json'

        for destination_file, path_dataset in Parse.PATHS.items():

            with open(path_dataset, encoding='utf8') as data_file:
                dataset = json.load(data_file)

            data = [{ 'question': [question['string'] for question in item['question'] if question['language'] == 'en'][0],
                      'id': item['id'],
                      'sparql_query_o': item['query']['sparql'],
                      'sparql_query': self.parse_sparql(item['query']['sparql']),
                      'entity mapping': [
                          {
                              'label':'',
                              'uri': '',
                           },
                      ],
                      'predicate mapping': [
                          {
                              'label':'',
                              'uri': '',
                           },
                      ],
                      } for item in dataset['questions']]

            with open(destination_file + "-annotated.json" , 'w') as f:
                json.dump(data, f, indent=1)

    def parse_sparql(self, question):
        prefixes = re.findall(f"PREFIX\s+(\w+:)\s+<([^>]+)>", question)
        prefixes_ = re.findall(f"PREFIX\s+\w+:\s+<[^>]+>", question)

        for prefix_ in prefixes_:
            question = question.replace(prefix_+' ','')

        for prefix in prefixes:
            question = question.replace(prefix[0],prefix[1])

        question = re.sub(r'((?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%:#]+)', r'<\1>', question)

        return question

if __name__ == "__main__":
    print('hello')

    parse = Parse()
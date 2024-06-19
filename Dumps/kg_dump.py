import bz2
import os
import pathlib
import re
import requests
from joblib import Parallel, delayed

from Dumps.spark_data import SparkData


class KGDump:

    def convert_to_cvs(self, path_file, f):  # , entities, relations):
        file = path_file
        extension_file = os.path.splitext(path_file)[1]
        if extension_file == '.bz2':
            source_file = bz2.BZ2File(file, "r")

            # Ignore the first line
            source_file.readline()
        elif extension_file == '.nt':
            source_file = open(file, "rb")

        # entities = set(entities + relations)

        countMatches = 0
        countLines = 0


        nextLine = '<s> <p> <o> .'.encode("utf-8")
        while (True):
            line = nextLine
            nextLine = source_file.readline()
            if len(line) == 0:
                print('Total Lines in ' + path_file + ':' + str(countLines))
                break
            countLines += 1
            try:
                txt = line.decode("utf-8")
            except:
                # print('except')
                txt = line.decode("iso_8859_1")
            originalLine = txt

            if '@' in txt:
                if not '@en' in txt:
                    continue

            txt = re.sub('\r?> <|> "', '"|"', txt)
            txt = re.sub('\r?<', '"', txt)
            txt = re.sub('\r?> .', '"', txt)
            #txt = re.sub('\r?@en .', '', txt)
            txt = re.sub('\r?\n', '', txt)

            countMatches += 1

            try:
                f.write(txt + "\n")
            except Exception as e:
                print(e)
                print(countLines)
                print(txt)
                print(originalLine)

    # def parse_files(self, path_file, f):  # , entities, relations):
    #     file = path_file
    #     source_file = bz2.BZ2File(file, "r")
    #
    #     # entities = set(entities + relations)
    #
    #     countMatches = 0
    #     countLines = 0
    #     while (True):
    #         line = source_file.readline()
    #         if len(line) == 0:
    #             print('Total Lines in ' + path_file + ':' + str(countLines))
    #             break
    #         countLines += 1
    #         try:
    #             txt = line.decode("utf-8")
    #         except:
    #             # print('except')
    #             txt = line.decode("iso_8859_1")
    #         originalLine = txt
    #
    #         if '@' in txt:
    #             if not '@en' in txt:
    #                 continue
    #
    #         txt = re.sub('\r?> <|> "', '|||', txt)
    #         txt = re.sub('\r?<', '', txt)
    #         txt = re.sub('\r?> .', '', txt)
    #         txt = re.sub('\r?\n', '', txt)
    #         parts_of_triple = txt.split('|||')
    #         countMatches += 1
    #
    #         try:
    #             f.write(originalLine)
    #         except Exception as e:
    #             print(e)
    #             print(countLines)
    #             print(txt)
    #             print(originalLine)

    def get_file(self, remote_file):  # , entities, relations):
        if '#' in remote_file:
            return

        if os.path.exists(SparkData.PATH_DUMP + remote_file.split('/')[-1] + ".ttl"):
            print("File " + remote_file.split('/')[-1] + " parsed.")
            return

        print("Downloading file " + remote_file)
        local_file = SparkData.PATH_DUMP + remote_file.split('/')[-1]
        if os.path.isfile(local_file):
            print('File exists')
        else:
            data = requests.get(remote_file)
            with open(local_file, 'wb')as file:
                try:
                    file.write(data.content)
                except:
                    print("fail")
        print("Parsing file " + local_file)

        if not os.path.exists(SparkData.PATH_DUMP + remote_file.split('/')[-1] + ".csv.bz2"):
        #try:
            #f = open(SparkData.PATH_DUMP + remote_file.split('/')[-1] + ".csv", "x", encoding="utf-8")
            f = bz2.open(SparkData.PATH_DUMP + remote_file.split('/')[-1] + ".csv.bz2", "wt", encoding="utf-8")
        #except  Exception as e:
        else:
            print("File " + remote_file.split('/')[-1] + " parsed.")
            return

        self.convert_to_cvs(local_file, f)  # , entities, relations)

        f.close()


    def __call__(self, *args, **kwargs):
        print('Parsing files started ... ')
        lines = pathlib.Path("./Dumps/dbpedia_files.txt").read_text().splitlines()
        Parallel(n_jobs=10)(delayed(self.get_file)(remote_file) for remote_file in lines if '#' not in remote_file)


 # def get_triples(sentence):
    #   sentence  = re.sub('\r?\n', ' ', sentence)
    #   sentence  = re.findall('\{.*?\}',sentence)[0][2:-1].rstrip(". ")  # Get substring inside { and }
    #   sentence = re.sub('\r? \. ', '|||', sentence)                     # Replace space_dot_space for |||
    #   sentence = re.sub('\r?\. <', '|||<', sentence)                    # Replace dot_< for |||
    #   sentence = re.sub('\r?\. \?', '|||?', sentence)                   # Replace dot_? for |||
    #   sentence = re.sub('\r?>|<', '', sentence)                         # Remove < OR >
    #   sentence = sentence.split('|||')
    #   return sentence

    # def load_entities():
    #   path_train = '../Datasets/LC-QuAD2.0/dataset/train.json'
    #   path_test = '../Datasets/LC-QuAD2.0/dataset/test.json'
    #
    #   with open(path_train) as data_file:
    #       train = json.load(data_file)
    #
    #   with open(path_test) as data_file:
    #       test = json.load(data_file)
    #
    #   all_dataset = train + test
    #
    #   entities = []
    #   relations = []
    #
    #   for triples in [get_triples(text['sparql_dbpedia18']) for text in all_dataset]:
    #     #print('--')
    #     for triple in triples:
    #       #print(triple)
    #       if not 'filter' in triple.lower():
    #         rdf_triple = triple.split(' ')
    #
    #         if 'http://' in rdf_triple[0]:
    #           if 'resource' in rdf_triple[0]:
    #             entities.append(rdf_triple[0])
    #           else:
    #             relations.append(rdf_triple[0])
    #
    #         if 'http://' in rdf_triple[2]:
    #           if 'resource' in rdf_triple[2]:
    #             entities.append(rdf_triple[2])
    #           else:
    #             relations.append(rdf_triple[2])
    #
    #         if 'http://' in rdf_triple[1]:
    #           relations.append(rdf_triple[1])
    #
    #   entities = list(set(entities))
    #   relations = list(set(relations))
    #   return entities, relations, all_dataset

#    print('Merging files ... ')
#    files = [path_dir + remote_file.split('/')[-1] + ".ttl" for remote_file in lines if '#' not in remote_file]
#  os.system("cat " + ' '.join(files) + " > " + path_dir + "dbpedia.ttl")
#   with open(path_dir + "dbpedia.ttl", 'w', encoding='utf-8') as outfile:
#     for fname in files:
#       print(fname)
#       with open(fname, encoding='utf-8') as infile:
#         #outfile.write(infile.read())
#         shutil.copyfileobj(infile, outfile)

# if __name__ == "__main__":
#     main()

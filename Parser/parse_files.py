import bz2
import os
import pathlib
import re
import requests
from joblib import Parallel, delayed

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

def parse_files(path_file, path_dir, f):  # , entities, relations):
    file = path_file
    source_file = bz2.BZ2File(file, "r")

    # entities = set(entities + relations)

    countMatches = 0
    countLines = 0
    while (True):
        line = source_file.readline()
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
        # if countLines % 10000000 == 0:
        #  print('Lines:' + str(countLines))
        #  print('Matched:' + str(countMatches))

        if '@' in txt:
            if not '@en' in txt:
                continue

        txt = re.sub('\r?> <|> "', '|||', txt)
        txt = re.sub('\r?<', '', txt)
        txt = re.sub('\r?> .', '', txt)
        txt = re.sub('\r?\n', '', txt)
        parts_of_triple = txt.split('|||')
        countMatches += 1

        # try:
        #   if path_file.split('/')[-1] == 'mappingbased-properties-reified.ttl.bz2':
        #     if parts_of_triple[2] in entities:
        #       countMatches+=1
        #     else:
        #       continue
        #   else:
        #     if parts_of_triple[0] in entities or parts_of_triple[1] in entities or parts_of_triple[2] in entities:
        #       countMatches+=1
        #     else:
        #       continue
        # except:
        #   print(countLines)
        #   print(txt)
        #   print(originalLine)

        try:
            f.write(originalLine)
        except Exception as e:
            print(e)
            print(countLines)
            print(txt)
            print(originalLine)


def get_file(remote_file, path_dir):  # , entities, relations):
    try:
        f = open(path_dir + remote_file.split('/')[-1] + ".ttl", "x", encoding="utf-8")
    except:
        print("File " + remote_file.split('/')[-1] + " parsed.")
        return

    if '#' in remote_file:
        return
    print("Downloading file " + remote_file)
    local_file = path_dir + remote_file.split('/')[-1]
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
    parse_files(local_file, path_dir, f)  # , entities, relations)

    f.close()


def main():
    print('Parsing files started ... ')
    # entities, relations, all = load_entities()

    path_files = "dbpedia_files.txt"
    path_dir = "../"

    lines = pathlib.Path(path_files).read_text().splitlines()
    #get_file(lines[0], path_dir)  # , entities, relations)
    Parallel(n_jobs=6)(delayed(get_file)(remote_file, path_dir) for remote_file in lines if '#' not in remote_file)


#    print('Merging files ... ')
#    files = [path_dir + remote_file.split('/')[-1] + ".ttl" for remote_file in lines if '#' not in remote_file]
#  os.system("cat " + ' '.join(files) + " > " + path_dir + "dbpedia.ttl")
#   with open(path_dir + "dbpedia.ttl", 'w', encoding='utf-8') as outfile:
#     for fname in files:
#       print(fname)
#       with open(fname, encoding='utf-8') as infile:
#         #outfile.write(infile.read())
#         shutil.copyfileobj(infile, outfile)

if __name__ == "__main__":
    main()

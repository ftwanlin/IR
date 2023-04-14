"""
    N19DCCN055: Nguyễn Tuấn Hiệp
    N19DCCN099: Tạ Quang Linh
    N19DCCN136: Trần Hoàng Phi
"""

import numpy as np
from functools import reduce
import string
from nltk.corpus import stopwords
import math
import re
# nltk.download('stopwords')


class Preprocessing:
    def __init__(self):
        self.reverted_index = self.create_reverted_index(self.remove_stop_words(
            self.docs_processing(self.read_file())))

    def read_file(self):
        docs = None
        with open('dataset/doc-text.txt', 'r') as f:
            docs = f.readlines()

            f.close()
        return docs

    def remove_stop_words(self, docs):
        english_stop_words = stopwords.words('english')
        new_docs = []
        for doc in docs:
            new_docs.append(
                ' '.join([word for word in doc.split() if word not in english_stop_words]))
        return new_docs

    def docs_processing(self, docs):
        docs_after_process = []
        # removing digit and break line ('\n)
        st1_pr_docs = list(filter(lambda y: not y.isdigit(),
                                  map(lambda x: x[:-1], docs)))
        # splitting allow forward flash
        table_remove_punctuation = str.maketrans(
            dict.fromkeys(string.punctuation))
        docs_tmp = []
        for doc in st1_pr_docs:
            if doc.endswith('/'):
                docs_after_process.append(' '.join(docs_tmp))
                docs_tmp.clear()
            else:
                docs_tmp.append(
                    ' '.join([w.lower().translate(table_remove_punctuation) for w in doc.split()]))
        return docs_after_process

    def create_reverted_index(self, processed_docs):
        reverted_index = reduce(lambda acc, word: {**acc, word: []}, list(
            set([word for word in (' '.join(processed_docs)).split() if word.strip()])), {})

        for i, doc in enumerate(processed_docs):
            for word in list(set([word for word in doc.split() if doc.strip()])):
                reverted_index[word].append(i + 1)

        for key in reverted_index:
            reverted_index[key] = list(set(reverted_index[key]))
            reverted_index[key].sort()

        list_tmp = list(reverted_index.items())
        list_tmp.sort(key=lambda x: x[0])

        reverted_index = dict(list_tmp)
        return reverted_index

    def post_listing_compress(self):
        my_list = self.reverted_index

        for key, value in my_list.items():
            compress = []
            compress.append(value[0])
            for i in range(1, len(value)):
                compress.append(value[i] - value[i-1])

            my_list[key] = compress

        return my_list


class VBCode:
    def __init__(self, n):
        self.base = n

    def read(self):
        docs = None
        with open('result/vbcode_encoding.txt', 'r') as f:
            docs = f.readlines()

            f.close()
        return self.processing(docs)

    def processing(self, docs):
        dict = {}
        for line in docs:
            line = line.split(' ')
            list = []
            tmp = []
            for i in range(1, len(line) - 1):
                tmp.append(line[i])
                if line[i][0] == '1':
                    list.append(tmp)
                    tmp = []

            # '\n' in the last element
            tmp.append(line[-1][:-1])
            list.append(tmp)

            dict[line[0]] = list
        # return dict: key(word), value(encode)
        return dict

    def extract_posting_list(self, posting_list):
        list = []
        list.append(posting_list[0])
        for i in range(1, len(posting_list)):
            list.append(list[i-1] + posting_list[i])

        return list

    def dec_to_bin(self, n):
        byte = bin(n)[2:]
        while len(byte) < self.base:
            byte = '0' + byte
        return byte

    def encode_number(self, n):
        byte = []
        baseNum = 1 << (self.base - 1)
        while True:
            byte.insert(0, n % (baseNum))
            if n < baseNum:
                break
            n = n // baseNum
        byte[len(byte)-1] += baseNum

        byte = [self.dec_to_bin(x) for x in byte]
        return byte

    def encode(self, numbers):
        bytestream = []
        for n in numbers:
            bytes = self.encode_number(int(n))
            bytestream.append(bytes)

        return bytestream

    def decode_number(self, byte):
        bytestream = ''

        for b in byte:
            bytestream += b[1:]
        return int(bytestream, 2)

    def decode(self, byte):
        posting_list = [self.decode_number(x) for x in byte]
        return posting_list

    def write_encode(self, posting_list):
        p = posting_list

        with open('result/vbcode_encoding.txt', 'w') as f:
            for key, value in p.items():
                value = self.encode(value)
                v = []
                for x in value:
                    for y in x:
                        v.append(y)
                f.write(key + ' ' + ' '.join(map(lambda x: str(x), v)) + '\n')

    def write_decode(self, posting_list):
        p = posting_list

        with open('result/vbcode_decoding.txt', 'w') as f:
            for key, value in p.items():
                value = self.extract_posting_list(self.decode(value))
                f.write(key + ' ' + ' '.join(map(lambda x: str(x), value)) + '\n')


class UnaryCode():
    def encode(self, n):
        return '1' * n + '0'

    def decode(self, s):
        return len(s) - 1


class GammaCode(UnaryCode):
    def read(self):
        docs = None
        with open('result/gammacode_encoding.txt', 'r') as f:
            docs = f.readlines()

            f.close()

        # return dict: key(word), value(encode)
        return self.processing(docs)

    def processing(self, docs):
        dict = {}
        for line in docs:
            line = line.split(' ')
            list = []
            for i in range(1, len(line) - 1):
                list.append(line[i])
            # '\n' in the last element
            list.append(line[-1][:-1])
            dict[line[0]] = list
        return dict

    def extract_posting_list(self, posting_list):
        list = []
        list.append(posting_list[0])
        for i in range(1, len(posting_list)):
            list.append(list[i-1] + posting_list[i])

        return list

    def dec_to_bin(self, n):
        return bin(n)[2:]

    def offset(self, s):
        return s[1:]

    def encode_number(self, n):
        bin = self.dec_to_bin(n)
        offset = self.offset(bin)
        unary = super().encode(len(bin) - 1)
        return unary + offset

    def encode(self, numbers):
        return [self.encode_number(n) for n in numbers]

    def decode_number(self, s):
        flag = 0
        for i in range(len(s)):
            if s[i] == '0':
                flag = i
                break
        return '1' + s[flag+1:]

    def decode(self, numbers):
        return [int(self.decode_number(n), 2) for n in numbers]

    def write_decode(self, posting_list):
        p = posting_list

        with open('result/gammacode_decoding.txt', 'w') as f:
            for key, value in p.items():
                value = self.extract_posting_list(self.decode(value))

                f.write(key + ' ' + ' '.join(map(lambda x: str(x), value)) + '\n')

    def write_encode(self, posting_list):
        p = posting_list

        with open('result/gammacode_encoding.txt', 'w') as f:
            for key, value in p.items():
                value = self.encode(value)
                f.write(key + ' ' + ' '.join(map(lambda x: str(x), value)) + '\n')


if __name__ == "__main__":
    prep = Preprocessing()
    data = prep.post_listing_compress()

    vb = VBCode(8)
    vb.write_encode(data)
    d = vb.read()
    vb.write_decode(d)

    g = GammaCode()
    g.write_encode(data)
    d = g.read()
    g.write_decode(d)

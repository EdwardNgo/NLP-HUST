from ast import literal_eval
import pandas as pd
import json
from underthesea import word_tokenize
from pyvi import ViTokenizer, ViPosTagger
import re
import csv  


def preprocess(document):
    # tách từ
    document = ViTokenizer.tokenize(document)
    # đưa về lower
    document = document.lower()
    # xóa các ký tự không can thiết
    document = re.sub(r'[^\s\wáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ_]',' ',document)
    # xóa khoảng trắng thừa
    document = re.sub(r'\s+', ' ', document).strip()
    return document


def convert_data_to_span_format(path):
    with open(path, 'r') as jf:
        records =[json.loads(record) for record in jf.read().split('\n')[0:-1]]
    iob_format = {}
    for idx in range(len(records)):
        total_length = len(records[idx]['text'])
        labels = records[idx]['labels']
        iob_format['sentence-{}'.format(idx)] = []
        for label_idx in range(len(labels)):
            # print()
            current_label_start = labels[label_idx][0]
            previous_label_end = labels[label_idx-1][1]
            current_label_end = labels[label_idx][1]
            if label_idx > 0 and label_idx != len(records) - 1:

                if current_label_start - previous_label_end > 0:
                    tokenize_text = preprocess(records[idx]['text'][previous_label_end:current_label_start])
                    iob_format['sentence-{}'.format(idx)].append({'text':tokenize_text, 'aspect':'0'})
            elif label_idx == len(labels) - 1 and current_label_end < total_length:
                    tokenize_text = preprocess(records[idx]['text'][current_label_end:total_length + 1])
                    iob_format['sentence-{}'.format(idx)].append({'text':tokenize_text, 'aspect':'0'})
                
            tokenize_text = preprocess(records[idx]['text'][current_label_start:current_label_end])
            iob_format['sentence-{}'.format(idx)].append({'text': tokenize_text, 'aspect':labels[label_idx][2]})
    return iob_format


def tag_iob_format(iob_format, filepath):
    f = open(filepath, 'a')
    writer = csv.writer(f)
    writer.writerow(['Sentence #', 'Word', 'Tag'])
    for key in iob_format:
        for record in iob_format[key]:
            text = record['text']
            aspect = record['aspect'].split('#')[0]
            words = text.split(' ')
            for idx in range(len(words)):
                if(words[idx] == ''): continue
                if idx == 0 and aspect != '0': 
                    writer.writerow([key, words[idx], 'B-' + aspect])
                elif idx != 0 and aspect != '0':
                    writer.writerow([key, words[idx], 'I-' + aspect])
                else:
                    writer.writerow([key, words[idx], aspect])
                    
    f.close()
    

def tag_iob_format_conll(iob_format, filepath):
    f = open(filepath, 'a')
    for key in iob_format:
        for record in iob_format[key]:
            text = record['text']
            aspect = record['aspect'].split('#')[0]
            words = text.split(' ')
            for idx in range(len(words)):
                if(words[idx] == ''): continue
                if idx == 0 and aspect != '0': 
                    f.write(words[idx] + ' ' + 'B-' + aspect + '\n')
                elif idx != 0 and aspect != '0':
                    f.write(words[idx] + ' ' + 'I-' + aspect + '\n')
                else:
                    f.write(words[idx] + ' ' + aspect + '\n')
        f.write('\n')
                    
    f.close()
                
                
if __name__ == "__main__":
#     prepare_flair_format('data/train.jsonl', 'flair_data/train.txt')
#     prepare_flair_format('data/dev.jsonl', 'flair_data/dev.txt')
#     prepare_flair_format('data/test.jsonl', 'flair_data/test.txt')
    # iob_format = convert_data_to_span_format('data/train.jsonl')
    # tag_iob_format_conll(iob_format, 'train2.conll')
    with open('dev.conll', 'r') as f:
        data = f.read()
        _data = re.sub(r'[\r\n][\r\n]{2,}', '\n\n', data)

        with open('dev_1.conll', 'w') as f_:
            f_.write(_data)
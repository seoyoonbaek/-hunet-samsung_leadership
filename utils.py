# -*- coding: utf-8 -*-
# -*- coding: euc-kr -*-
################################### utils.py ###################################

import os
import re
import json
import glob
import time
from collections import Counter
import xml.etree.ElementTree as elemTree
import pandas as pd


def read_xml(xml_path: str) -> object:
    """
    :param xml_path: xml 파일의 경로
    :return: root: xml 파일의 최상위 root
    """
    xml_file = open(xml_path, 'rt', encoding='UTF8')
    tree = elemTree.parse(xml_file)
    root = tree.getroot()
    return root


def read_json(json_file_path: str) -> object:
    """
    :param json_file_path: json 파일의 경로
    :return: 파일 데이터
    """
    with open(json_file_path, 'r', encoding='UTF8') as fp:
        json_data = json.load(fp)
    return json_data


def json_str_to_dict(json_data: object) -> dict:
    """
    :param json_data: json_data 파일 데이터
    :return: 파일 데이터를 dictionary 형태로 변환
    """
    json_data_dict = json.loads(json_data)
    return json_data_dict


def read_json_to_dict(json_file_path: str) -> dict:
    """
    :param json_file_path: json 파일의 경로
    :return: dictionary 형태로 변환된 json file 데이터
    """
    json_data = read_json(json_file_path)
    dict_data = json_str_to_dict(json_data)
    return dict_data


def get_file_path_list(dir_name: str) -> list:
    """
    :param dir_name: directory 이름
    :return: directory 속 file들의 경로 list
    """
    INPUT_DATA_DIR = os.path.join(os.path.abspath(''), 'data')
    dir_path = os.path.join(INPUT_DATA_DIR, 'ability_keyword_book_data/korean-dict-nikl', dir_name)
    filetPath_list = []
    for filetPath in sorted(glob.glob(dir_path + '/*.xml')):
        filetPath_list.append(filetPath)
    return filetPath_list


def save_dict_to_json(dict_data: object, json_path: object) -> None:
    """
    :param dict_data: dictionary 형태의 데이터
    :param json_path: 저장 파일 경로
    """
    # CONVERT dictionary to json using json.dump
    json_data = json.dumps(dict_data, ensure_ascii=False)
    json_path = str(json_path) + str('.json')
    with open(json_path, 'w', encoding='UTF-8') as file:
        file.write(json.dumps(json_data))


def save_ability_keyword_book(dict_data: dict, save_json_dir_path: str) -> None:
    """
    :param dict_data: dictionary 형태의 데이터
    :param save_json_dir_path: json파일을 저장할 directory path
    """
    file_name = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(save_json_dir_path, file_name)
    save_dict_to_json(dict_data, json_path)
    print('saved_file_name : ', str(file_name) + str('.json'))


def analyse_cnt(dictionary_data: dict) -> None:
    """
    :param dictionary_data: count 분석을 하고자 하는 dictionary data
    """
    cnt = 0
    for k, v_list in dictionary_data.items():
        cnt += len(v_list)
        print('len of [{}] : {}'.format(k, len(v_list)))
    print('----------------------')
    print('total : {}'.format(cnt))
    print()


def extract_korean(word: str) -> str:
    """
    :param word: string 단어
    :return: string 단어 속 한글만 추출하여 반환
    """
    hangul = re.compile('[^ ㄱ-힣+]+')
    preprocessed_word = hangul.sub('', word)  # 제거
    return preprocessed_word


def extract_english(word: str) -> str:
    pattern_eng = "[^-a-zA-Z]"
    """
    :param word: string 단어
    :return: string 단어 속 영어만 추출하여 반환
    """
    english = re.compile(pattern_eng)
    preprocessed_word = english.sub('', word)  # 제거
    return preprocessed_word


# def print_mode_helper() -> None:
#     # 각 mode에 대한 사용법을 출력한다.
#     print('ability_keyword_book MODE를 선택해주세요.')
#     print('--C [csv_file] : create ability_keyword_book using csv_file')
#     print('--U [csv_file] [json_file] : update ability_keyword_book(=json_file) using csv_file')
#     print('--L [json_file] : load ability_keyword_book(=json_file)')

#####################################################################################
def count_line(paragraph_list: list) -> list:
    """
    입력 받은 문장의 수를 파악하기 위한 함수
    :param paragraph_list: 파악 대상
    :return:
    """
    sentence_log = []
    for paragraph in paragraph_list:
        paragraph_length = len(paragraph.splitlines()) / 2
        sentence_log.append(paragraph_length)

    return sentence_log


def sum_list(token_list: list) -> list:
    """
    이중 리스트를 하나의 리스트로 변환해주는 함수
    :param token_list: 이중 리스트
    :return:
    """
    result = sum(token_list, [])
    return result


def combine_name_count(leader_list: list, sent_log: list, num: int) -> list:
    combine_list = []
    for i in range(num):
        temp = []
        temp.append(leader_list[i])
        temp.append(sent_log[i])
        combine_list.append(temp)
    return combine_list


def get_leader_id(data_pd: pd.DataFrame) -> list:
    """
    대상 리더들의 id list를 반환
    :param data_pd: 리더 id가 포함된 pandas dataframe
    :return:
    """
    combine_pd = data_pd.groupby('KnoxID')['답변내용'].agg(','.join).reset_index()
    leader_list = list(combine_pd['KnoxID'])
    paragraph_list = list(combine_pd['답변내용'])
    leader_num = len(list(combine_pd['KnoxID']))
    sentence_num = count_line(paragraph_list)
    combine_list = combine_name_count(leader_list, sentence_num, leader_num)

    return combine_list


def count_token(tokens: list) -> dict:
    """
    토큰 리스트를 {'token':num...}의 사전으로 변환 시켜주는 함수
    :param tokens: 토큰 리스트들
    :return: 변환된 dictionary
    """
    counter = Counter(tokens)
    token_dict = dict(counter.most_common())
    return token_dict


def is_not_empty(target_list):
    """
    비어있지 않은 리스트인지를 확인하는 함수
    """
    if len(target_list) == 0:
        return 0
    else:
        return 1

# -*- coding: utf-8 -*-
# -*- coding: euc-kr -*-
################################### eng_augmented_ability_keyword_book.py ###################################
import os
import sys

import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from khaiii import KhaiiiApi
from LEA.utils import *

DATA_DIR = os.path.join(os.path.abspath(''), 'input')

def get_list_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        data.append(file.readlines())
    eng_words = list(filter(None, re.split('[,\'\" ]', data[0][0])))
    return eng_words


def get_translated_keyword_kor_eng_dict_list(eng_words):
    translator = Translator()
    translator.raise_Exception = True

    keyword_kor_eng_dict_list = {}
    for eng_word in tqdm(eng_words):
        kor_word = translator.translate(eng_word, src='en', dest='ko').text
        if kor_word in keyword_kor_eng_dict_list.keys():
            keyword_kor_eng_dict_list[kor_word].append(eng_word)
        else:
            keyword_kor_eng_dict_list[kor_word] = [eng_word]

    #print(len(keyword_kor_eng_dict_list))
    return keyword_kor_eng_dict_list


def split_dict_by_filter(filter_func, df):
    filtered_dict = {}
    no_filtered_dict = {}
    for ko, en in df.iloc():
        if filter_func(ko):
            filtered_dict[ko] = en
        else:
            no_filtered_dict[ko] = en
    #print('filtered_dict : ', len(filtered_dict))
    #print('no_filtered_dict : ', len(no_filtered_dict))
    return filtered_dict, no_filtered_dict


def get_filtered_dict_by_filter(keyword_kor_eng_dict_list, filter_func):
    kor_eng_df = pd.DataFrame({'kor': keyword_kor_eng_dict_list.keys(), 'eng': keyword_kor_eng_dict_list.values()})
    in_eng_dict, only_kor_dict = split_dict_by_filter(filter_func, kor_eng_df)
    return in_eng_dict, only_kor_dict


def get_only_kor_dict(eng_words_translated_dict_list):
    def in_eng_filter(word):
        if extract_english(word):
            return True
        else:
            return False

    def in_kor_filter(word):
        if extract_korean(word):
            return True
        else:
            return False

    in_eng_dict, no_eng_dict = get_filtered_dict_by_filter(eng_words_translated_dict_list, in_eng_filter)
    in_kor_dict, no_kor_dict = get_filtered_dict_by_filter(no_eng_dict, in_kor_filter)

    return in_kor_dict


def get_word_pos_tuple(tokenizer: object, x: str) -> list:
    """
    :param tokenizer: tokenizer
    :param x: 토큰화할 단어
    :return: [토큰화된 단어, 품사] 정보를 담은 list
    """
    word_pos_list = []
    analyzed = tokenizer.analyze(x)
    for word in analyzed:
        for morph in word.morphs:
            word_pos_list.append((morph.lex, morph.tag))
    return word_pos_list


def get_unique_refined_dict(refined_dict):
    for k, v_list in refined_dict.items():
        refined_dict[k] = list(set(v_list))
    return refined_dict


def get_only_nouns(ko_pos_list):
    noun_refined_ko = []
    for ko_pos in ko_pos_list:
        # 명사만
        if ko_pos[1] in ['NNG', 'NNP', 'XR']:
            noun_refined_ko.append(ko_pos[0])

    return noun_refined_ko


def get_refined_dict(noun_refined_ko, en_list):
    refined_dict = {}
    for t in noun_refined_ko:
        # 토큰 각각
        if len(t) > 1:
            if t in refined_dict.keys():
                refined_dict[t] += en_list
            else:
                refined_dict[t] = en_list

    # 토큰을 다 합친 거
    if len(noun_refined_ko) > 1:
        temp_join = ''.join(noun_refined_ko)
        if temp_join in refined_dict.keys():
            refined_dict[temp_join] += en_list
        else:
            refined_dict[temp_join] = en_list

    return refined_dict


def get_refined_ability_keyword_dict(ability_keyword_dict):
    # 명사만 -> k를 토큰 각각, 다 합친 거 하나 ->  len(k)>1
    tokenizer = KhaiiiApi()
    refined_dict = {}
    
    for ko, en_list in ability_keyword_dict.items():
        ko_pos_list = get_word_pos_tuple(tokenizer, ko)
            
        noun_refined_ko = get_only_nouns(ko_pos_list)
        
        if noun_refined_ko:
            temp_refined_dict = get_refined_dict(noun_refined_ko, en_list)
            for temp_k, temp_v_list in temp_refined_dict.items():
                if temp_k in refined_dict.keys():
                    refined_dict[temp_k] = refined_dict[temp_k]+temp_v_list
                else:
                    refined_dict[temp_k] = temp_v_list
                
            #refined_dict.update(get_refined_dict(noun_refined_ko, en_list))

    unique_refined_dict = get_unique_refined_dict(refined_dict)
    
    return unique_refined_dict

def get_augmented_ability_keyword_book_not1(origin_augmented_ability_keyword_book):
    augmented_ability_keyword_book_not1 ={}
    for k in origin_augmented_ability_keyword_book.keys():
        augmented_ability_keyword_book_not1[k] = list(filter(lambda x: len(x)>1, origin_augmented_ability_keyword_book[k]))

    #analyse_cnt(augmented_ability_keyword_book_not1)
    return augmented_ability_keyword_book_not1

def get_en_ability_keyword_book(augmented_ability_keyword_book_not1, refined_only_kor_dict_list):
    en_ability_keyword_book = {}

    for k, v_list in augmented_ability_keyword_book_not1.items():
        temp_augmented_list = []
        for v in v_list:
            if v in refined_only_kor_dict_list.keys():
                temp_augmented_list+=refined_only_kor_dict_list[v]
        en_ability_keyword_book[k] = temp_augmented_list

    return en_ability_keyword_book

def get_en_augmented_ability_keyword_book_unique(en_augmented_ability_keyword_book):
    en_augmented_ability_keyword_book_unique = {}
    for k, v_list in en_augmented_ability_keyword_book.items():
        en_augmented_ability_keyword_book_unique[k] = list(set(v_list))

    #analyse_cnt(en_augmented_ability_keyword_book_unique)
    return en_augmented_ability_keyword_book_unique


def concat_en_augmented_ability_keyword_book(augmented_ability_keyword_book_not1, en_ability_keyword_book):
    en_augmented_ability_keyword_book = {}
    for k, v_list in augmented_ability_keyword_book_not1.items():
        en_augmented_ability_keyword_book[k] = v_list + en_ability_keyword_book[k]

    en_augmented_ability_keyword_book_unique = get_en_augmented_ability_keyword_book_unique(
        en_augmented_ability_keyword_book)
    return en_augmented_ability_keyword_book, en_augmented_ability_keyword_book_unique


def get_en_augmented_ability_keyword_book(origin_ability_keyword_book):
    print('origin_ability_keyword_book 분석 : ')
    analyse_cnt(origin_ability_keyword_book)
    
    #eng_words = get_list_from_txt(os.path.join(DATA_DIR, 'tokenized_leadership_eng.txt'))  # 3341
    #googletrans
    #keyword_kor_eng_dict_list = get_translated_keyword_kor_eng_dict_list(eng_words)
    
    keyword_kor_eng_dict_list = read_json_to_dict(os.path.join(DATA_DIR, 'keyword_kor_eng_dict.json'))
    #print('keyword_kor_eng_dict_list : ', len(keyword_kor_eng_dict_list))


    #제대로 번역된 단어들만 남기기(filter)
    only_kor_dict = get_only_kor_dict(keyword_kor_eng_dict_list)
    #print('only_kor_dict : ', len(only_kor_dict)) #2057

    #번역된 kor에 대한 토큰화 및 증강
    #한국어 정제 -> 명사 위주 + 1글자 제거 + 중복제거
    refined_only_kor_dict_list = get_refined_ability_keyword_dict(only_kor_dict)
    
    augmented_ability_keyword_book_not1 = get_augmented_ability_keyword_book_not1(origin_ability_keyword_book)

    #en_ability_keyword_book 생성
    en_ability_keyword_book = get_en_ability_keyword_book(augmented_ability_keyword_book_not1, refined_only_kor_dict_list)

    #기존 증강된 북 + en_ability_keyword_book
    en_augmented_ability_keyword_book, en_augmented_ability_keyword_book_unique = concat_en_augmented_ability_keyword_book(
        augmented_ability_keyword_book_not1, en_ability_keyword_book)

    #기술 단어들로 '기술력' 역량 추가
    tech_words = get_list_from_txt(os.path.join(DATA_DIR, 'tech_words.txt')) # 684
    en_augmented_ability_keyword_book_6ability = en_augmented_ability_keyword_book_unique.copy()
    en_augmented_ability_keyword_book_6ability['기술력'] = tech_words

    print('en_augmented_ability_keyword_book 분석 : ')
    analyse_cnt(en_augmented_ability_keyword_book_6ability)
    
    save_dict_to_json(en_augmented_ability_keyword_book_6ability, os.path.join(os.path.abspath(''), 'output/en_augmented_ability_keyword_book'))
    
    return en_augmented_ability_keyword_book_6ability

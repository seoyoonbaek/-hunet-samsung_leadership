# -*- coding: utf-8 -*-
# -*- coding: euc-kr -*-
################################### eng_augmented_ability_keyword_book.py ###################################
import pandas as pd
from tqdm import tqdm
from googletrans import Translator
from khaiii import KhaiiiApi
#from LEA.utils import *
from utils import *

DATA_DIR = os.path.join(os.path.abspath(''), 'input')

def get_list_from_eng_txt(file_path: str) -> list:
    # englisth word data를 담은 txt file을 list type으로 변환합니다.
    """

    Args:
        file_path: eng_txt file path

    Returns: txt file의 내용을 담은 list

    """
    data = []
    with open(file_path, 'r') as file:
        data.append(file.readlines())
    eng_words = list(filter(None, re.split('[,\'\" ]', data[0][0])))
    return eng_words
#
#
# def get_translated_keyword_kor_eng_dict_list(eng_words):
#     """
#
#     Args:
#         eng_words:
#
#     Returns:
#
#     """
#     translator = Translator()
#     translator.raise_Exception = True
#
#     keyword_kor_eng_dict_list = {}
#     for eng_word in tqdm(eng_words):
#         kor_word = translator.translate(eng_word, src='en', dest='ko').text
#         if kor_word in keyword_kor_eng_dict_list.keys():
#             keyword_kor_eng_dict_list[kor_word].append(eng_word)
#         else:
#             keyword_kor_eng_dict_list[kor_word] = [eng_word]
#
#     # print(len(keyword_kor_eng_dict_list))
#     return keyword_kor_eng_dict_list


def split_dict_by_filter(filter_func: object, df: pd.DataFrame) -> dict: #function
    # DataFrame에 filter_func을 적용하여 dict형태의 결과를 반환한다.
    """

    Args:
        filter_func: filter 내용을 담은 function
        df: filter를 적용할 대상 DataFrame

    Returns: 대상 DataFrame에 대해 filter가 적용된 결과물, 적용되지 않은 결과물을 dict형태로 반환

    """
    filtered_dict = {}
    no_filtered_dict = {}
    for ko, en in df.iloc():
        if filter_func(ko):
            filtered_dict[ko] = en
        else:
            no_filtered_dict[ko] = en
    return filtered_dict, no_filtered_dict


def get_filtered_dict_by_filter(keyword_kor_eng_list_dict: dict, filter_func: object) -> dict:
    # keyword_kor_eng_list_dict에 filter를 적용한 결과 반환
    """

    Args:
        keyword_kor_eng_list_dict: keyword token들의 한영 번역 dict {한국어:[해석된 영어들], ..}
        filter_func: filter 내용을 담은 function

    Returns: keyword_kor_eng_list_dict에 filter_func을 적용한 결과 dict 반환

    """
    kor_eng_df = pd.DataFrame({'kor': keyword_kor_eng_list_dict.keys(), 'eng': keyword_kor_eng_list_dict.values()})
    in_eng_dict, only_kor_dict = split_dict_by_filter(filter_func, kor_eng_df)
    return in_eng_dict, only_kor_dict


def get_only_kor_dict(keyword_kor_eng_list_dict: dict) -> dict:
    # 영한 번역이 올바로 적용된 data 선별
    """

    Args:
        keyword_kor_eng_list_dict: keyword token들의 한영 번역 dict {한국어:[해석된 영어들], ..}

    Returns: keyword_kor_eng_list_dict 에서 영한 번역이 올바로 적용된 data만 dict로 반환

    """
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

    in_eng_dict, no_eng_dict = get_filtered_dict_by_filter(keyword_kor_eng_list_dict, in_eng_filter)
    in_kor_dict, no_kor_dict = get_filtered_dict_by_filter(no_eng_dict, in_kor_filter)

    return in_kor_dict


def get_word_pos_tuple(tokenizer: object, x: str) -> list:
    # word에 대해 토큰화된 품사 정보 list 반환
    """

    Args:
        tokenizer: tokenizer
        x: 토큰화할 단어

    Returns: [토큰화된 단어, 품사] tuple을 담은 list

    """

    word_pos_list = []
    analyzed = tokenizer.analyze(x)
    for word in analyzed:
        for morph in word.morphs:
            word_pos_list.append((morph.lex, morph.tag))
    return word_pos_list


def get_unique_refined_dict(refined_dict: dict) -> dict:
    # dict의 각 key에 대한 value(list)의 중복을 제거
    """

    Args:
        refined_dict:

    Returns:

    """
    for k, v_list in refined_dict.items():
        refined_dict[k] = list(set(v_list))
    return refined_dict


def get_only_nouns(ko_pos_list: list) -> list:
    # 명사인 token만을 반환
    """

    Args:
        ko_pos_list: token에 대한 (lex, tag) tuple을 담은 list

    Returns: ko_pos_list 중 pos가 명사인 단어만을 담은 list

    """
    noun_refined_ko = []
    for ko_pos in ko_pos_list:
        # 명사만
        if ko_pos[1] in ['NNG', 'NNP', 'XR']:
            noun_refined_ko.append(ko_pos[0])

    return noun_refined_ko


def get_augmented_dict(noun_refined_ko: object, en_list: object) -> object:
    # 명사 토큰화된 한국어 keyword로 키워드 증강
    """

    Args:
        noun_refined_ko: keyword token 중 명사 만을 담은 list
        en_list: noun_refined_ko의 번역된 영어를 담은 list

    Returns: {한국어 keyword token: [해당 변역 englisth words]}

    """
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


def get_refined_ability_keyword_dict(only_kor_dict: object) -> object:
    ## 번역된 kor에 대한 토큰화 및 정제
    # 한국어 정제 -> 명사 위주 + 중복제거
    """
    Args:
        only_kor_dict: 한영 변환이 제대로 이루어진 한영 변역을 담은 dict {한국어:[해석된 영어들], ..}

    Returns: 토큰화 및 정제가 적용된

    """
    # 명사만 -> k를 토큰 각각, 다 합친 거 하나 ->  len(k)>1
    tokenizer = KhaiiiApi()
    refined_dict = {}

    for ko, en_list in only_kor_dict.items():
        # 토큰화
        ko_pos_list = get_word_pos_tuple(tokenizer, ko)

        # 명사만
        noun_refined_ko = get_only_nouns(ko_pos_list)

        if noun_refined_ko:
            # 영어 단어들로 keyword book 증강

            # {한국어 token:해당 영어 list} 생성
            augmented_dict = get_augmented_dict(noun_refined_ko, en_list)

            # keyword book 증강
            for temp_k, temp_v_list in augmented_dict.items():
                if temp_k in refined_dict.keys():
                    refined_dict[temp_k] = refined_dict[temp_k] + temp_v_list
                else:
                    refined_dict[temp_k] = temp_v_list

    # 중복제거
    unique_refined_dict = get_unique_refined_dict(refined_dict)

    return unique_refined_dict


def get_augmented_ability_keyword_book_not1(origin_augmented_ability_keyword_book: object) -> object:
    # 기존 ability_keyword_book에서 한글자 단어 제거
    """

    Args:
        origin_augmented_ability_keyword_book: 기존 ability_keyword_book

    Returns: 한글자 단어를 제거한 ability_keyword_book

    """
    augmented_ability_keyword_book_not1 = {}
    for k in origin_augmented_ability_keyword_book.keys():
        augmented_ability_keyword_book_not1[k] = list(
            filter(lambda x: len(x) > 1, origin_augmented_ability_keyword_book[k]))

    # analyse_cnt(augmented_ability_keyword_book_not1)
    return augmented_ability_keyword_book_not1


def get_en_ability_keyword_book(augmented_ability_keyword_book_not1: object, refined_only_kor_dict_list: object) -> object:

    """

    Args:
        augmented_ability_keyword_book_not1:
        refined_only_kor_dict_list:

    Returns:

    """
    en_ability_keyword_book = {}

    for k, v_list in augmented_ability_keyword_book_not1.items():
        temp_augmented_list = []
        for v in v_list:
            if v in refined_only_kor_dict_list.keys():
                temp_augmented_list += refined_only_kor_dict_list[v]
        en_ability_keyword_book[k] = temp_augmented_list

    return en_ability_keyword_book


def get_en_augmented_ability_keyword_book_unique(en_augmented_ability_keyword_book: object) -> object:

    """

    Args:
        en_augmented_ability_keyword_book:

    Returns:

    """
    en_augmented_ability_keyword_book_unique = {}
    for k, v_list in en_augmented_ability_keyword_book.items():
        en_augmented_ability_keyword_book_unique[k] = list(set(v_list))

    # analyse_cnt(en_augmented_ability_keyword_book_unique)
    return en_augmented_ability_keyword_book_unique


def concat_en_augmented_ability_keyword_book(augmented_ability_keyword_book_not1: object, en_ability_keyword_book: object) -> object:
    """

    Args:
        augmented_ability_keyword_book_not1:
        en_ability_keyword_book:

    Returns:

    """
    en_augmented_ability_keyword_book = {}
    for k, v_list in augmented_ability_keyword_book_not1.items():
        en_augmented_ability_keyword_book[k] = v_list + en_ability_keyword_book[k]

    en_augmented_ability_keyword_book_unique = get_en_augmented_ability_keyword_book_unique(
        en_augmented_ability_keyword_book)
    return en_augmented_ability_keyword_book, en_augmented_ability_keyword_book_unique


def get_en_augmented_ability_keyword_book(origin_ability_keyword_book):
    """

    Args:
        origin_ability_keyword_book: 기존 ability_keyword_book

    Returns: 영어단어 및 기술단어가 증강된 ability_keyword_book

    """
    print('origin_ability_keyword_book 분석 : ')
    analyse_cnt(origin_ability_keyword_book)

    ##삭제예정
    # eng_words = get_list_from_txt(os.path.join(DATA_DIR, 'tokenized_leadership_eng.txt'))  # 3341
    # googletrans
    # keyword_kor_eng_dict_list = get_translated_keyword_kor_eng_dict_list(eng_words)

    # token {한국어:[해석된 영어들], ..}
    keyword_kor_eng_list_dict = read_json_to_dict(os.path.join(DATA_DIR, 'keyword_kor_eng_dict.json'))

    # 제대로 번역된 단어들만 남기기(filter)
    only_kor_dict = get_only_kor_dict(keyword_kor_eng_list_dict) #2057

    ## 번역된 kor에 대한 토큰화 및 정제
    # 한국어 정제 -> 명사 위주 + 중복제거
    refined_only_kor_dict_list = get_refined_ability_keyword_dict(only_kor_dict)

    # 기존 origin_ability_keyword_book에서 1글자 제거
    augmented_ability_keyword_book_not1 = get_augmented_ability_keyword_book_not1(origin_ability_keyword_book)

    # 증강된 en_ability_keyword_book 생성
    en_ability_keyword_book = get_en_ability_keyword_book(augmented_ability_keyword_book_not1,
                                                          refined_only_kor_dict_list)
    # 기존 증강된 북 + en_ability_keyword_book
    en_augmented_ability_keyword_book, en_augmented_ability_keyword_book_unique = concat_en_augmented_ability_keyword_book(
        augmented_ability_keyword_book_not1, en_ability_keyword_book)

    # 기술 단어들로 '기술력' 역량 추가
    tech_words = get_list_from_eng_txt(os.path.join(DATA_DIR, 'tech_words.txt'))  # 684
    en_augmented_ability_keyword_book_6ability = en_augmented_ability_keyword_book_unique.copy()
    en_augmented_ability_keyword_book_6ability['기술력'] = tech_words

    #결과 print
    print('en_augmented_ability_keyword_book 분석 : ')
    analyse_cnt(en_augmented_ability_keyword_book_6ability)

    save_dict_to_json(en_augmented_ability_keyword_book_6ability,
                      os.path.join(os.path.abspath(''), 'output/en_augmented_ability_keyword_book'))

    return en_augmented_ability_keyword_book_6ability

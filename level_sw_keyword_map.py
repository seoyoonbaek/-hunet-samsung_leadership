# -*- coding: utf-8 -*-
# -*- coding: euc-kr -*-
################################### level_sw_keyword_map.py ###################################

import re
import os.path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import font_manager, rc

ability_list = ['ImmersionBoost', 'FutureInsight', 'Performance', 'HRGrowth', 'SelfDev']

def set_global_variables(N):
    global category_names
    category_names = ['top ' + str(i) for i in range(1, N + 1)]

    global kor_font
    kor_font = font_manager.FontProperties(fname='/home/hunetdb/src/dev.tm.lee/LEA/input/MaruBuri-Regular.ttf')


def cleaning_tokens(token_list):
    return list(filter(None, re.split('[\[\"\', \]]', token_list)))


def get_cleaned_df(df):
    info_dataframe = df.copy()
    info_dataframe['강점토큰'] = list(map(cleaning_tokens, df['강점토큰']))
    info_dataframe['보완점토큰'] = list(map(cleaning_tokens, df['보완점토큰']))
    return info_dataframe


def get_level_tokens_dict(ability, df, sw):
    level_tokens_dict = {}
    for level in range(1, 11):
        level_tokens = df[df[ability] == level][sw]  # w_tokens
        temp_total = []
        for tokens in level_tokens:
            temp_total += tokens
        level_tokens_dict[level] = temp_total
    return level_tokens_dict


def get_level_tokens_dict_sw(df, sw):
    level_tokens_dict_sw = {}
    for ability in ability_list:
        level_tokens_dict_sw[ability] = get_level_tokens_dict(ability, df, sw)
    return level_tokens_dict_sw


def get_topn_per_level(level_tokens_dict, N):
    topn_level_tokens = {}
    for ability, level_tokens in level_tokens_dict.items():
        temp_topn_list = []
        for level, tokens_list in level_tokens.items():
            temp_topn_list.append(Counter(tokens_list).most_common()[:N])
        topn_level_tokens[ability] = temp_topn_list
    return topn_level_tokens


def get_np_data(results):
    data = []
    for x in results.values():
        data.append(list(k[1] for k in x))
    return np.array(data)


def set_level_keyword_chart(ax, results):
    labels = list(results.keys())
    data = get_np_data(results)
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('YlGn')(
        np.linspace(0.15, 0.85, data.shape[1]))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (col_name, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        label_data = []
        for x in results.values():
            label_data.append(list(k for k in x)[i])

        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=col_name, color=color)
        x_centers = starts + widths / 2
        plt.yticks(fontsize=11)

        r, g, b, _ = color
        text_color = 'black'
        for y, (x, c) in enumerate(zip(x_centers, label_data)):
            ax.text(x, y, str(c[0]),
                    ha='center', va='center',
                    color=text_color, fontproperties=kor_font, fontsize=10)


def create_level_keyword_charts(results2020, results2021):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    plt.figure(figsize=(15, 7))

    ax = plt.subplot(1, 2, 1)
    ax.set_title("2020", fontsize=10)
    set_level_keyword_chart(ax, results2020)

    ax = plt.subplot(1, 2, 2)
    ax.set_title("2021~2022", fontsize=10)
    set_level_keyword_chart(ax, results2021)


def print_level_topn_plot(level_tokens_dict2020, level_tokens_dict2021, N, sw_hint):
    sw_title = '[강점] ' if sw_hint == 's' else '[보완점] '

    topn_level_tokens_dict2020 = get_topn_per_level(level_tokens_dict2020, N)
    topn_level_tokens_dict2021 = get_topn_per_level(level_tokens_dict2021, N)

    for ability in ability_list:
        results2020 = {str('level ' + str(i + 1)): v_list for i, v_list in
                       enumerate(topn_level_tokens_dict2020[ability])}
        results2021 = {str('level ' + str(i + 1)): v_list for i, v_list in
                       enumerate(topn_level_tokens_dict2021[ability])}

        create_level_keyword_charts(results2020, results2021)

        plt.suptitle(str(sw_title) + str(ability), fontsize=20, fontproperties=kor_font)

        #plot 저장
        plt.savefig(os.path.join('output/level_keyword_chart', ''.join([sw_hint, '_', ability, '.png'])))


def leadership_level_topn_plot(df_list, N):
    # global_variables setting
    set_global_variables(N)

    level_tokens_dict_list_s = []
    level_tokens_dict_list_w = []

    # df -> 강점 list, 보완점 list
    for df in df_list:
        level_tokens_dict_s = get_level_tokens_dict_sw(df, '강점토큰')
        level_tokens_dict_w = get_level_tokens_dict_sw(df, '보완점토큰')

        level_tokens_dict_list_s.append(level_tokens_dict_s)
        level_tokens_dict_list_w.append(level_tokens_dict_w)

    print_level_topn_plot(*level_tokens_dict_list_s, N, 's')
    print_level_topn_plot(*level_tokens_dict_list_w, N, 'w')

    print('All level keyword charts are saved !')


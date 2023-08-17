from typing import Dict

# https://github.com/NJUNLP/GTS#results=, ASTE-Data-V2-dataset
# https://arxiv.org/pdf/2010.04640.pdf
gts: Dict = {'info': f'Results taken from: https://arxiv.org/pdf/2010.04640.pdf',
             '14lap': {'Precision': 58.54, 'Recall': 50.65, 'F1': 54.3},
             '14res': {'Precision': 68.71, 'Recall': 67.67, 'F1': 68.17},
             '15res': {'Precision': 60.69, 'Recall': 60.54, 'F1': 60.61},
             '16res': {'Precision': 67.39, 'Recall': 66.73, 'F1': 67.06}}

# https://aclanthology.org/2021.acl-long.367.pdf, ASTE-Data-V2-dataset
span_based: Dict = {'info': 'Results taken from: https://aclanthology.org/2021.acl-long.367.pdf',
                    '14lap': {'Precision': 63.44, 'Recall': 55.84, 'F1': 59.38},
                    '14res': {'Precision': 72.89, 'Recall': 70.89, 'F1': 71.85},
                    '15res': {'Precision': 62.18, 'Recall': 64.45, 'F1': 63.27},
                    '16res': {'Precision': 69.45, 'Recall': 71.17, 'F1': 70.26}}

# https://arxiv.org/pdf/2102.08549v3.pdf, ASTE-Data-V2-dataset
first_to_then_polarity: Dict = {'info': 'https://arxiv.org/pdf/2102.08549v3.pdf',
                                '14lap': {'Precision': 57.84, 'Recall': 59.33, 'F1': 58.58},
                                '14res': {'Precision': 63.59, 'Recall': 73.44, 'F1': 68.16},
                                '15res': {'Precision': 54.53, 'Recall': 63.3, 'F1': 58.59},
                                '16res': {'Precision': 63.57, 'Recall': 71.98, 'F1': 67.52}}

# https://aclanthology.org/2021.acl-short.64.pdf, ASTE-Data-V2-dataset
t5: Dict = {'info': 'https://aclanthology.org/2021.acl-short.64.pdf',
            '14lap': {'Precision': None, 'Recall': None, 'F1': 60.78},
            '14res': {'Precision': None, 'Recall': None, 'F1': 72.16},
            '15res': {'Precision': None, 'Recall': None, 'F1': 62.1},
            '16res': {'Precision': None, 'Recall': None, 'F1': 70.1}}

# https://arxiv.org/pdf/2010.02609.pdf, ASTE-Data-V2-dataset
jet_t: Dict = {'info': 'https://arxiv.org/pdf/2010.02609.pdf',
               '14lap': {'Precision': 53.53, 'Recall': 43.28, 'F1': 47.86},
               '14res': {'Precision': 63.44, 'Recall': 54.12, 'F1': 58.41},
               '15res': {'Precision': 68.2, 'Recall': 42.89, 'F1': 52.66},
               '16res': {'Precision': 65.28, 'Recall': 51.95, 'F1': 57.85}}

# https://arxiv.org/pdf/2010.02609.pdf, ASTE-Data-V2-dataset
jet_o: Dict = {'info': 'https://arxiv.org/pdf/2010.02609.pdf',
               '14lap': {'Precision': 55.39, 'Recall': 47.33, 'F1': 51.04},
               '14res': {'Precision': 70.56, 'Recall': 55.94, 'F1': 62.4},
               '15res': {'Precision': 64.45, 'Recall': 51.96, 'F1': 57.53},
               '16res': {'Precision': 70.42, 'Recall': 58.37, 'F1': 63.83}}

# https://arxiv.org/pdf/2103.15255.pdf, ASTE-Data-V2-dataset
more_fine_grained: Dict = {'info': 'https://arxiv.org/pdf/2103.15255.pdf',
                           '14lap': {'Precision': 56.6, 'Recall': 55.1, 'F1': 55.8},
                           '14res': {'Precision': 69.3, 'Recall': 69.0, 'F1': 69.2},
                           '15res': {'Precision': 55.8, 'Recall': 61.5, 'F1': 58.5},
                           '16res': {'Precision': 61.2, 'Recall': 72.7, 'F1': 66.5}}

# https://openreview.net/pdf?id=Z9vIuaFlIXx, ASTE-Data-V2-dataset
sambert: Dict = {'info': 'https://openreview.net/pdf?id=Z9vIuaFlIXx',
                 '14lap': {'Precision': 62.26, 'Recall': 59.15, 'F1': 60.66},
                 '14res': {'Precision': 70.29, 'Recall': 74.92, 'F1': 72.53},
                 '15res': {'Precision': 65.12, 'Recall': 63.51, 'F1': 64.3},
                 '16res': {'Precision': 68.01, 'Recall': 75.44, 'F1': 71.53}}

# https://arxiv.org/pdf/2204.12674.pdf, ASTE-Data-V2-dataset
SBC: Dict = {'info': 'https://arxiv.org/pdf/2204.12674.pdf',
             '14lap': {'Precision': 63.64, 'Recall': 61.80, 'F1': 62.71},
             '14res': {'Precision': 77.09, 'Recall': 70.99, 'F1': 73.92},
             '15res': {'Precision': 63.00, 'Recall': 64.95, 'F1': 63.96},
             '16res': {'Precision': 75.20, 'Recall': 71.40, 'F1': 73.25}}

# @inproceedings{naglik_lango,
#     place = {Osaka},
#     booktitle = {Proceedings of the Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2023)},
#     title = {Exploiting Phrase Interrelations in Span-level Neural Approaches for Aspect Sentiment Triplet Extraction},
#     author = {Naglik, Iwo and Lango, Mateusz},
#     year = "2023"
# }
SpanASTE: Dict = {'info': 'https://arxiv.org/pdf/2204.12674.pdf',
                  '14lap': {'Precision': 66.98, 'Recall': 60.55, 'F1': 63.56},
                  '14res': {'Precision': 75.29, 'Recall': 72.56, 'F1': 73.89},
                  '15res': {'Precision': 66.44, 'Recall': 64.74, 'F1': 65.54},
                  '16res': {'Precision': 71.12, 'Recall': 72.45, 'F1': 71.77}}

OTHER_RESULTS: Dict = {
    'gts': gts,
    'span_based': span_based,
    'first_to_then_polarity': first_to_then_polarity,
    't5': t5,
    'jet_t': jet_t,
    'jet_o': jet_o,
    'more_fine_grained': more_fine_grained,
    'sambert': sambert,
    'SBC': SBC,
    'SpanASTE': SpanASTE
}

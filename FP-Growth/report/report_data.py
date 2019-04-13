import time

from algorithm.apriori import Apriori
from algorithm.fpGrowth import FPGrowth
MINSUP_DICT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
# MINSUP_DICT = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
EXP_TIMES = 1

# algorithm_dict = {
#     'fpg': FPGrowth,
#     'apr': Apriori
# }
algorithm_dict = {'fpg': FPGrowth}
input_file_path = '../data/upt_en.txt'
output_file_path = './report.csv'


def str_to_list(input_data):
    data_list = [
        data.replace(',\n', '').split(',')
        for data in input_data
    ]
    return data_list


def count_time(func, data_list, minsup):
    duration = 0
    for count in range(EXP_TIMES):
        start_time = time.time()
        fp_results = func(data_list, minsup)
        end_time = time.time()
        duration += (end_time - start_time)
    duration /= EXP_TIMES

    fp_num = len(fp_results.fp_dict.keys())

    return duration, fp_num


with open(output_file_path, 'w') as output:
    output.write('minsup,fp_num,algorithm,time\n')

    with open(input_file_path, 'r') as input_file:

        data_list = str_to_list(input_file)
        data_length = len(data_list)

        for key in algorithm_dict:
            for current_minsup in MINSUP_DICT:
                (fp_duration, fp_num) = count_time(algorithm_dict[key], data_list, int(current_minsup * data_length))
                data = [str(current_minsup), str(fp_num), key, str(fp_duration)]
                print(data)

                output.write(','.join(data))
                output.write('\n')

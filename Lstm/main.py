import time
import os
from core.model import Model
from core.utils import Utils


def test():
    # 开始计时
    start = time.clock()
    test_date = time.time()
    data_path = './data/processed/prt_mfd_7.csv'
    save_path = './test/purchase/'
    col_start = 1
    col_end = 4
    train_begin = 0
    train_end = 397
    time_step = 15
    rnn_unit = 10
    lr = 0.0006
    run_times = 100
# ---------------------------------------------
    if not os.path.exists(save_path):
        os.makedirs(save_path)
# ---------------------------------------------
    model = Model(save_path, test_date, data_path, col_start, col_end, train_begin, train_end, time_step, rnn_unit, lr, run_times)
    test_y, test_predict = model.get_test_result()
# -------------------------------
    end = time.clock()
    spend = end - start
    utils = Utils()
    test_score = utils.get_score(test_y, test_predict)
    describe = 'purchase(1308_1408,[purchase,redeem,mfd_7])'
    utils.save_test(test_y, test_predict, save_path, test_date, lr, run_times, time_step, rnn_unit, spend, test_score, describe)


if __name__ == '__main__':
    test()

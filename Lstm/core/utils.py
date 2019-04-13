import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
class Utils:

	def get_score(self, test_y, test_predict):
		test_error = np.divide(np.abs(test_y - test_predict.flatten()), test_y)
		test_score = 0
		for item in test_error:
			print(item)
			if item <= 0.3:
				test_score += 10 - (100.0 / 3) * item
		return test_score

	def save_test(self, test_y, test_predict,save_path,test_date,lr, run_times, time_step,rnn_unit, spend,test_score,describe):
		text_file = open(save_path+"score.txt", "a+")
		text_file.write(
			'test_date:%f, lr:%f, times:%d, time_step:%d, unit:%d, spend:%f, score:%f, describe:%s,' % (test_date,
																				  lr,
																				  run_times,time_step, rnn_unit,
																				  spend,
																				  test_score,describe))
		text_file.write('\n')
		text_file.close()
		df_predict = pd.DataFrame(test_predict)
		# 导出结果
		df_predict.to_csv(
			save_path +str(test_date)+"/predict" + ".csv", header=0)

		# 以折线图表示结果
		plt.figure()
		print("预测结果大小:", len(test_predict))
		plt.plot(list(range(len(test_predict))), test_predict, color='b')
		plt.plot(list(range(len(test_y))), test_y, color='r')
		plt.savefig(save_path +str(test_date) + '/test.png')
		plt.show()






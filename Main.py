# import numpy as np
#
# class DataManager:
#     training_test_percentage = 0.0

from Model import Model
from datamanipulator import Manager


class Main:

    def run(self):
        manager = Manager()
        model = Model(manager)
        model.set_data(manager)
        model.set_class_target()
        model.gradient_descent(5)
        model.regression.get_best_for_class()
        # class_error = regression.get_best_for_class()
        # print(class_error)
        return


obj = Main()
obj.run()

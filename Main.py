# import numpy as np
#
# class DataManager:
#     training_test_percentage = 0.0

from Model import Model
from datamanipulator import Manager

class Main:

    def run(self):
        manager = Manager()
        model = Model(manager, 10)
        model.estimate()
        model.regression.get_best_for_class()
        print("###################")
        model.classification.best_classification()

        # class_error = regression.get_best_for_class()
        # print(class_error)
        return


obj = Main()
obj.run()

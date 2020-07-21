# import numpy as np
#
# class DataManager:
#     training_test_percentage = 0.0

from Model import Model
from datamanipulator import Manager


class Main:

    def run(self):
        manager = Manager()
        model = Model(manager, 5)
        model.estimate()
        model.regression.get_best_for_class()
        return


obj = Main()
obj.run()

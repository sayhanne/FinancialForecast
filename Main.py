# import numpy as np
#
# class DataManager:
#     training_test_percentage = 0.0

from Model import LR
from datamanipulator import Manager
from HannesTool import HannesTool


class Main:

    def run(self):
        manager = Manager()
        regression = LR()
        regression.getData(manager)
        regression.gradientDescent(5)
        return


obj = Main()
obj.run()

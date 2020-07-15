#import numpy as np
#
# class DataManager:
#     training_test_percentage = 0.0
from Model import LR
from DataManipulater import Manager
from HannesTool import HannesTool
class Main:
    # init
    
    data_manager = Manager(25)
    model = LR(data_manager.get_number_of_pieces())
    tool = HannesTool()
    W = model.miniBatchGradientDescent(data_manager)
    print("W:", W)
    print("Error:",tool.get_err('Class', W))



# from HannesTool import HannesTool
import random
# import statistics


# class DataCollector:
#
#     def __init__(self, manager):
#         self.x = None
#         self.y = None
#         self.manager_object = manager
#         self.set_next_piece()
#
#     def set_next_piece(self):  # must return X matrix and Y array
#         self.x = []
#         self.y = []
#         for piece in self.manager_object.get_training_piece():
#             self.x.append(piece[0])
#             self.y.append(piece[1])
#
#     def get_current_piece(self):
#         return self.x, self.y


class LR:

    def __init__(self, number):  # constructor
        self.weightArrays = []
        self.numberOfPredictors = 0
        self.parameters = []
        self.target = []
        self.managerObject = None
        self.estimation = []
        self.data = []
        self.learning_rate = 0.00000000005
        self.batch_number = number

    def initializeParameters(self):
        arr = []
        for i in range(self.numberOfPredictors + 1):  # adding initial intercept and weights of predictors
            arr.append(random.random())
        return arr

    def initializeEstimation(self):
        arr = []
        for i in range(len(self.target)):
            arr.append(0.0)
        return arr

    def estimationForRow(self, row):
        estimation = self.parameters[0]  # first the intercept is added
        for i in range(len(self.parameters) - 1):
            estimation += self.parameters[i+1] * self.data[row][i]   # then all the remaining is added
        return estimation

    def estimationCalculator(self):
        for i in range(len(self.target)):
            self.estimation[i] = self.estimationForRow(i)
        return

    def costFunction(self, predictor):  # cost function is mse
        self.estimationCalculator()
        cost = 0.0
        for i in range(len(self.data)):
            if predictor == 0:
                cost += (self.target[i] - self.estimation[i]) * -1
            else:
                cost += (self.target[i] - self.estimation[i]) * -self.data[i][predictor-1]
        return 2 * cost / len(self.data)

    def miniBatchGradientDescent(self, manager, numOfPredictors):
        self.numberOfPredictors = numOfPredictors  # number of predictors can change
        self.managerObject = manager
        converged = False
        index = 0
        count = 0
        self.parameters = self.initializeParameters()
        while not converged:
            self.data, self.target = self.getDataPiece()
            self.estimation = self.initializeEstimation()
            for i in range(numOfPredictors + 1):
                self.parameters[i] = self.parameters[i] - self.learning_rate * self.costFunction(i)
            index += 1
            if index == self.batch_number:
                index = 0
                count += 1
                if count == 150:
                    converged = True
                    self.weightArrays.append(self.parameters)
                    break
                else:
                    continue
            else:
                continue
        return

    def getDataPiece(self):  # must return X matrix and Y array
        x = []
        y = []
        for piece in self.managerObject.get_training_piece():
            x.append(piece[0])
            y.append(piece[1])
        return x, y

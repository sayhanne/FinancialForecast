from HannesTool import HannesTool
import random
import math


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

    def __init__(self):  # constructor
        self.pieceNumber = 0
        self.tool = HannesTool()
        self.data_BigTrain = None
        self.data_BigTest = None
        self.weightArrays = []
        self.mseArray = []
        self.numberOfPredictors = 0
        self.parameters = []
        self.estimation = []
        self.estimation_test = []
        self.data = None
        self.target = None
        self.data_test = None
        self.target_test = None
        self.index = None
        self.learning_rate = 0.000005

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
            estimation += self.parameters[i + 1] * self.data[row][i]  # then all the remaining is added
        return estimation

    def estimationForRowTest(self, row):
        estimation = self.parameters[0]  # first the intercept is added
        for i in range(len(self.parameters) - 1):
            estimation += self.parameters[i + 1] * self.data_test[row][i]  # then all the remaining is added
        return estimation

    def estimationCalculator(self):
        for i in range(len(self.target)):
            self.estimation[i] = self.estimationForRow(i)
        return

    def estimationCalculatorTest(self):
        for i in range(len(self.target_test)):
            self.estimation_test[i] = self.estimationForRowTest(i)
        return

    def costFunction(self, predictor):  # cost function is mse
        self.estimationCalculator()
        cost = 0.0
        for i in range(len(self.data)):
            if predictor == 0:
                cost += (self.target[i] - self.estimation[i]) * -1
            else:
                cost += (self.target[i] - self.estimation[i]) * -self.data[i][predictor - 1]
        return 2 * cost / len(self.data)

    def rmseForTrainPiece(self):
        mse = 0.0
        self.estimationCalculator()
        for i in range(len(self.data)):
            mse += (self.target[i] - self.estimation[i]) ** 2

        return math.sqrt(mse / len(self.data))

    def rmseForTestPiece(self):
        mse = 0.0
        self.estimationCalculatorTest()
        for i in range(len(self.data_test)):
            mse += (self.target_test[i] - self.estimation_test[i]) ** 2

        return math.sqrt(mse / len(self.data_test))

    def gradientDescent(self, numOfPredictors):
        self.numberOfPredictors = numOfPredictors  # number of predictors can change
        completed = False
        count = 0
        while not completed:
            self.data, self.target, self.data_test, self.target_test = self.getPiece()
            self.estimation = self.initializeEstimation()
            self.estimation_test = self.initializeEstimation()
            self.parameters = self.initializeParameters()
            mse_train = self.rmseForTrainPiece()
            min_mse = mse_train
            converged = False
            print("calculating")
            while not converged:
                for j in range(self.numberOfPredictors + 1):
                    self.parameters[j] = self.parameters[j] - self.learning_rate * self.costFunction(j)
                mse_train = self.rmseForTrainPiece()
                if min_mse > mse_train > 2.5:
                    # print(min_mse)
                    min_mse = mse_train
                else:
                    converged = True
            print("final weights", self.parameters)
            print(self.rmseForTestPiece())
            print(self.target_test)
            print(self.estimation_test)
            mse = 0.0
            min_mse = 0.0
            # print(self.tool.get_err("Class", self.parameters))
            self.weightArrays.append(self.parameters)
            self.mseArray.append(self.rmseForTestPiece())
            count += 1
            if count == self.pieceNumber:
                completed = True

        return self.weightArrays, self.mseArray

    def getPiece(self):
        piece = []
        test_piece = []
        x = None
        y = []
        x_test = None
        y_test = []
        for i in range(self.index * 16, (self.index + 1) * 16):
            piece.append(self.data_BigTrain[i])
        for i in range(self.index * 8, (self.index + 1) * 8):
            test_piece.append(self.data_BigTest[i])
        y = self.getColumn(piece, self.numberOfPredictors)  # last column
        y_test = self.getColumn(test_piece, self.numberOfPredictors)  # last column
        for row in piece:
            del row[self.numberOfPredictors]
        for row in test_piece:
            del row[self.numberOfPredictors]

        x = piece
        x_test = test_piece
        self.index += 1
        return x, y, x_test, y_test

    def getColumn(self, matrix, i):
        return [row[i] for row in matrix]

    def getData(self, manager):
        self.data_BigTrain = manager.data_training
        self.data_BigTest = manager.data_test
        self.index = 0
        self.pieceNumber = len(self.data_BigTest) / 8

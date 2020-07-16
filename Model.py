from HannesTool import HannesTool
from sklearn import linear_model
import math
class LR:
    def __init__(self):  # constructor
        self.pieceNumber = 0
        self.lr = None
        self.parameters = []
        self.lasso = None
        self.tool = HannesTool()
        self.data_BigTrain = None
        self.data_BigTest = None
        self.weightArrays = []
        self.mseArray = []
        self.numberOfPredictors = 0
        self.data = None
        self.target = None
        self.data_test = None
        self.target_test = None
        self.estimationForTest = None
        self.index = None

    def rmse(self):
        mse = 0.0
        for i in range(len(self.target_test)):
            mse += (self.target_test[i] - self.estimationForTest[i]) ** 2
        return math.sqrt(mse / len(self.target_test))

    def gradientDescent(self, numOfPredictors):
        self.numberOfPredictors = numOfPredictors  # number of predictors can change
        completed = False
        count = 0
        while not completed:
            self.data, self.target, self.data_test, self.target_test = self.getPiece()
            self.lr = linear_model.LinearRegression()
            self.lr.fit(self.data, self.target)
            self.estimationForTest = self.lr.predict(self.data_test)
            rmse = self.rmse()
            print("iteration-->", count + 1)
            print("intercept-->", self.lr.intercept_)
            print("coefficients-->", self.lr.coef_)
            self.parameters.append(self.lr.intercept_)  # intercept
            for i in range(len(self.lr.coef_)):         # weights
                self.parameters.append(self.lr.coef_[i])
            self.weightArrays.append(self.parameters)
            self.mseArray.append(rmse)
            print("rmse-->", rmse)
            print("weights", self.parameters)
            print("*********")
            self.parameters = []
            count += 1
            if count == self.pieceNumber:
                completed = True
        return self.weightArrays, self.mseArray

    def getData(self, manager):
        self.data_BigTrain = manager.data_training
        self.data_BigTest = manager.data_test
        self.lasso = linear_model.Lasso()
        self.index = 0
        self.pieceNumber = len(self.data_BigTest) / 8

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

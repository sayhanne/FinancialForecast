from hannestool import HannesTool
from sklearn import linear_model
import math

class Model:
    def __init__(self, manager):
        self.data_BigTrain = None
        self.data_BigTest = None
        self.tr_target_class = []
        self.te_target_class = []
        self.set_data(manager)
        self.set_class_target()
        self.regression = LR(self.data_BigTrain, self.data_BigTest)

    def gradient_descent(self, predictors_number):
        """gradient descent icin LR'a müraacaat eder"""
        self.regression.gradient_descent(predictors_number)

    def set_data(self, manager):
        self.data_BigTrain = manager.data_training
        self.data_BigTest = manager.data_test
        self.index = 0
        self.piece_number = len(self.data_BigTest) / 8

    def set_class_target(self):
        for row in self.data_BigTrain:
            if row[-1] > 0:
                self.tr_target_class.append(1)
            else:
                self.tr_target_class.append(0)
        for row in self.data_BigTest:
            if row[-1] > 0:
                self.te_target_class.append(1)
            else:
                self.te_target_class.append(0)
##########################################################################################
##########################################################################################
##########################################################################################
class LR:
    """Lineer regression yapması için sınıf"""
    def __init__(self, data_tr, data_te):  # constructor
        self.data_BigTrain = data_tr
        self.data_BigTest = data_te
        self.piece_number = 0
        self.lr = None
        self.parameters = []
        self.lasso = None
        self.lassoParams = []
        self.tool = HannesTool()
        self.weightArrays = []
        self.numberOfPredictors = 0
        self.data = None
        self.target = None
        self.data_test = None
        self.target_test = None
        self.estimationForTest = None
        self.index = None
        self.index = 0
        self.piece_number = len(self.data_BigTest) / 8

    def rmse(self):
        mse = 0.0
        for i in range(len(self.target_test)):
            mse += (self.target_test[i] - self.estimationForTest[i]) ** 2
        return math.sqrt(mse / len(self.target_test))

    def gradient_descent(self, numOfPredictors):
        self.numberOfPredictors = numOfPredictors  # number of predictors can change
        completed = False
        count = 0
        while not completed:
            print("iteration-->", count + 1)
            self.data, self.target, self.data_test, self.target_test = self.getPiece()
            self.lr = linear_model.LinearRegression()
            self.lasso = linear_model.Lasso()
            self.lr.fit(self.data, self.target)
            self.lasso.fit(self.data, self.target)
            self.estimationForTest = self.lr.predict(self.data_test)
            rmse = self.rmse()
            self.estimationForTest = self.lasso.predict(self.data_test)
            rmse2 = self.rmse()

            self.parameters.append(self.lr.intercept_)  # intercept
            for i in range(len(self.lr.coef_)):  # weights
                self.parameters.append(self.lr.coef_[i])

            self.lassoParams.append(self.lasso.intercept_)  # intercept
            for i in range(len(self.lasso.coef_)):  # weights
                self.lassoParams.append(self.lasso.coef_[i])

            for i in range(len(self.parameters)):
                if self.lassoParams == 0.0:
                    self.parameters[i] = 0.0

            self.weightArrays.append(self.parameters)
            self.weightArrays.append(self.lassoParams)
            print("rmse-->", rmse)
            print("rmse lasso-->", rmse2)
            print("*********")
            self.parameters = []
            self.lassoParams = []
            count += 1
            if count == self.piece_number:
                print("TRAINING DONE")
                completed = True
        return self.weightArrays

    

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

    # def get_correctness(self):
    #     errors = []
    #     for weight_index in range(len(self.weightArrays)):
    #         test_data = self.data_BigTest[(weight_index * 8): ((weight_index + 1) * 8)]
    #         weights = self.weightArrays[weight_index]
    #         #print(weights, weight_index, test_data, "\n\n")
    #         error = HannesTool().get_err("Class", weights, test_data)
    #         errors.append(error)
    #     print("errors:", errors)
    #     return sum(errors)/len(errors)

    def get_best_for_class(self):
        errors = []
        max_index = 0
        estimation = []
        target = []
        max_correctness = 0
        max_estimation = []
        for weight_index in range(len(self.weightArrays)):
            weights = self.weightArrays[weight_index]
            # print(weights, weight_index, test_data, "\n\n")
            correctness, estimation, target = HannesTool().get_err("Class", weights, self.data_BigTest)
            print("correctness of weight array", weight_index, correctness)
            if max_correctness < correctness:
                max_estimation = estimation
                max_index = weight_index
                max_correctness = correctness
            errors.append(correctness)
        print("test target", target)
        print("best estimation", max_estimation)
        print("best", self.weightArrays[max_index], "correctness:", max_correctness)
        return sum(errors) / len(errors)

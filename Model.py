"""Makine öğreniminin model bölümü"""
import math
from sklearn import linear_model
from hannestool import HannesTool


class Model:
    """Modülün ana kısmı"""
    def __init__(self, manager, predictor):
        self.manager = manager
        self.piece_number = len(self.manager.data_test)/8
        self.regression = LR(self.manager.data_training, self.manager.data_test, self.piece_number, predictor)
        self.classification = Logistic(self.manager.data_training, self.manager.data_test, self.piece_number, predictor)

    def estimate(self):
        """tüm modeller burada çalışıyor"""
        self.regression.gradient_descent()
        print("#########")
        self.classification.classification()


##########################################################################################
##########################################################################################
##########################################################################################
class Logistic:
    """Logistic Regression yapan model"""
    def __init__(self, data_tr, data_te, number, predictor):  # constructor
        self.data_BigTrain = data_tr
        self.data_BigTest = data_te
        self.logistic = None
        self.data = None
        self.index = 0
        self.target = None
        self.data_test = None
        self.target_test = None
        self.estimationForTest = None
        self.piece_number = number
        self.numberOfPredictors = predictor

    def classification(self):
        completed = False
        count = 0
        while not completed:
            print("iteration-->", count + 1)
            self.data, self.target, self.data_test, self.target_test = self.getPiece()
            self.logistic = linear_model.LogisticRegression()
            self.logistic.fit(self.data, self.target)
            self.estimationForTest = self.logistic.predict(self.data_test)
            rmse = self.rmse()
            print("rmse for classification", rmse)
            print("*********")
            count += 1
            if count == self.piece_number:
                print("--------Classification done--------")
                completed = True
        return

    def rmse(self):
        mse = 0.0
        for i in range(len(self.target_test)):
            mse += (self.target_test[i] - self.estimationForTest[i]) ** 2
        return math.sqrt(mse / len(self.target_test))

    def set_class_target(self, train_y, test_y):
        y = []
        y_test = []
        for i in range(len(train_y)):
            if train_y[i] > 0:
                y.append(1)
            else:
                y.append(0)
        for i in range(len(test_y)):
            if test_y[i] > 0:
                y_test.append(1)
            else:
                y_test.append(0)
        return y, y_test

    def getPiece(self):
        piece = []
        index = 0
        test_piece = []
        x = None
        y = []
        x_test = None
        y_test = []
        for i in range(self.index * 16, (self.index + 1) * 16):
            piece.append([])
            for a in range(len(self.data_BigTrain[0])):
                piece[index].append(self.data_BigTrain[i][a])
            index += 1

        index = 0
        for i in range(self.index * 8, (self.index + 1) * 8):
            test_piece.append([])
            for a in range(len(self.data_BigTest[0])):
                test_piece[index].append(self.data_BigTest[i][a])
            index += 1

        y = self.getColumn(piece, self.numberOfPredictors)  # last column
        y_test = self.getColumn(test_piece, self.numberOfPredictors)  # last column

        y_class, y_class_test = self.set_class_target(y, y_test)
        for row in piece:
            del row[self.numberOfPredictors]
        for row in test_piece:
            del row[self.numberOfPredictors]

        x = piece
        x_test = test_piece
        self.index += 1
        return x, y_class, x_test, y_class_test

    def getColumn(self, matrix, i):
        return [row[i] for row in matrix]


class LR:
    """Linear Regression yapan model"""
    def __init__(self, data_tr, data_te, number, predictor):  # constructor
        self.data_BigTrain = data_tr
        self.data_BigTest = data_te
        self.lr = None
        self.parameters = []
        self.lasso = None
        self.lassoParams = []
        self.weightArrays = []
        self.numberOfPredictors = predictor
        self.data = None
        self.target = None
        self.data_test = None
        self.target_test = None
        self.estimationForTest = None
        self.index = 0
        self.piece_number = number

    def rmse(self):
        mse = 0.0
        for i in range(len(self.target_test)):
            mse += (self.target_test[i] - self.estimationForTest[i]) ** 2
        return math.sqrt(mse / len(self.target_test))

    def gradient_descent(self):
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
                if self.lassoParams[i] == 0.0:
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
                print("--------Regression done--------")
                completed = True
        return self.weightArrays

    def getPiece(self):
        piece = []
        test_piece = []
        index = 0
        x = None
        y = []
        x_test = None
        y_test = []
        for i in range(self.index * 16, (self.index + 1) * 16):
            piece.append([])
            for a in range(len(self.data_BigTrain[0])):
                piece[index].append(self.data_BigTrain[i][a])
            index += 1

        index = 0
        for i in range(self.index * 8, (self.index + 1) * 8):
            test_piece.append([])
            for a in range(len(self.data_BigTest[0])):
                test_piece[index].append(self.data_BigTest[i][a])
            index += 1

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
            # print("correctness of weight array", weight_index, correctness)
            if max_correctness < correctness:
                max_estimation = estimation
                max_index = weight_index
                max_correctness = correctness
            errors.append(correctness)
        print("test target", target)
        print("best estimation", max_estimation)
        print("best", self.weightArrays[max_index], "correctness:", max_correctness)
        return sum(errors) / len(errors)

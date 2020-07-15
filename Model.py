from HannesTool import HannesTool
# from DataManipulater import Manager
import random
import statistics

class DataCollector:

    x = None
    y = None
    manager_object = None

    def __init__(self, manager):
        global x, y, manager_object
        manager_object = manager
        self.set_next_piece()

    def set_next_piece(self):
        global x, y
        x = []
        y = []
        for piece in manager_object.get_training_piece():
            x.append(piece[0])
            y.append(piece[1])

    def get_current_piece(self):
        return x,y

class LR:
    # reader = None

    def __init__(self, number):  # constructor
        self.parameters = [0.5, 0.5]
        self.target = []
        self.managerObject = None
        self.estimation = []
        self.data = []
        self.learning_rate = 0.00000000005
        self.batch_number = number

    def initializeParameters(self):
        self.parameters[1] = random.random()/10
        self.parameters[0] = statistics.mean(self.data) * random.random() * 2

    def estimationCalculator(self):
        for i in range(len(self.target)):
            self.estimation[i] = (self.estimationForRow(i))
        return

    def estimationForRow(self, row):
        return self.parameters[0] + self.data[row] * self.parameters[1]

    def costFunction(self, predictor):  # cost function is mse
        self.estimationCalculator()
        cost = 0
        for i in range(len(self.data)):
            if predictor == 0:
                cost += (self.target[i] - self.estimation[i]) * -1
            else:
                cost += (self.target[i] - self.estimation[i]) * -self.data[i]
        return 2 * cost / len(self.data)

    def initializeEstimation(self):
        arr = []
        for i in range(len(self.target)):
            arr.append(None)
        return arr

    def mseForDataPiece(self):
        total = 0.0
        for i in range(len(self.target)):
            total += (self.target[i] - self.estimation[i]) ** 2
        return total / len(self.target)

    def miniBatchGradientDescent(self, manager):
        self.managerObject = manager
        initialized = False
        index = 1
        ina=0
        mse = 0.0
        converged = False
        tool = HannesTool()
        while not converged:
            self.data, self.target = self.getDataPiece()
            self.estimation = self.initializeEstimation()
            if not initialized:
                self.initializeParameters()
                initialized = True
                #print("W's first:", self.parameters)
                for i in range(len(self.data)):
                    print(self.data[i]*self.parameters[1]+self.parameters[0], end=' ')
                print()
                
            for j in range(2):
                cost = self.costFunction(j)
                #print("bgd",j,"parametreler:",self.parameters[j], "cost:",cost)
                self.parameters[j] = self.parameters[j] - self.learning_rate * cost


            index = index + 1
            if index > self.batch_number:
                index = 1
                # mse = tool.get_err("R2", self.parameters)
                mse = self.mseForDataPiece()
                if(ina<150):
                    ina+=1
                else:
                    return self.parameters



    def getDataPiece(self):
        x = []
        y = []
        for piece in self.managerObject.get_training_piece():
            x.append(piece[0])
            y.append(piece[1])
        return x, y

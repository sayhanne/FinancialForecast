"""Yardımcı Toollar için kullanılan bir modül"""
import matplotlib.pyplot as plt
import numpy as np

class HannesTool:
    """Tek classı"""

    def get_mean(self, target):
        """returns mean of target"""
        tot = 0
        for i in enumerate(target):
            tot += target[i]
        return tot / len(target)

    def get_tss(self, data_test):
        """returns TSS of target"""
        targets = []
        for _ in enumerate(data_test):
            targets.append(data_test[-1])
        mean = self.get_mean(targets)
        tss = 0
        for target in targets:
            tss += (target - mean) ** 2
        return tss

    def get_rss(self, weights, data_test):
        """returns RSS of model"""
        rss, tss = 0, 0
        for test_sample in data_test:
            # estimation yani fx oluşturuluyor
            estimation = 0
            for w_index in enumerate(weights):
                if w_index == 0:
                    estimation = weights[w_index]
                else:
                    estimation += test_sample[w_index - 1] * weights[w_index]
            # rss hesap kısmı
            target = test_sample[-1]
            tss += self.get_tss(target)
            rss += (target - estimation) ** 2
        return rss

    def get_r2(self, weights, data_test):
        """returns R2 of model"""
        return 1 - self.get_rss(weights, data_test) / self.get_tss(data_test)

    def get_err(self, error_type, weights, data_test):
        """returns error of model according to given type (data_test denilen şey 1 paket)"""
        if error_type == 'R2':
            return self.get_r2(weights, data_test)
        elif error_type == 'Class':
            return self.get_err_class(weights, data_test)

    def get_err_class(self, weights, data_test):
        targetArr = []
        estimationArr = []
        """tahminimizin class olarak doğruluk oranını hesaplıyor"""
        total = 0
        for test_sample in data_test:
            # estimation yani fx oluşturuluyor
            estimation = 0
            for w_index in enumerate(weights):
                if w_index[0] == 0:
                    estimation = w_index[1]
                else:
                    estimation += test_sample[w_index[0] - 1] * w_index[1]
            #### hesap kısmı
            target = test_sample[-1]
            targetArr.append(target)
            estimationArr.append(estimation)
            if target * estimation > 0:
                total += 1
        return total / len(data_test), estimationArr, targetArr

    def get_err_class_logistic(self, logistic, X, target):
        total = 0.0
        estimations = logistic.predict(X)
        est_arr = []
        for i in range(len(target)):
            if target[i] == estimations[i]:
                total += 1
        for i in estimations:
            est_arr.append(i)
        return total / len(target), est_arr

    def plot(self, estimations, target, index):
        if index == 0:
            plt.plot(estimations, label="estimations")
            plt.plot(target, label="targets")
        elif index == 1:
            plt.plot(estimations, label="estimations", linestyle="--")
            plt.plot(target, label="targets")
        plt.legend()
        plt.show()

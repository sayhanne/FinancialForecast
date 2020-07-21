"""csv ile uğraşılıp verinin alındığı modül"""
import pandas as pd
import matplotlib.pyplot as plt

######################
####Constants#########
######################
DATA_DEST = "Data\\BIST_100_Gecmis_Verileri_Haftalik.csv"
DATA_DEST_EX = "Data\\BIST_100_RSI.csv"
TOTAL_LENGTH = pd.read_csv(DATA_DEST).shape[0]
TR_DATA_LENGTH, TE_DATA_LENGTH = 16, 8
BOUND = TR_DATA_LENGTH + TE_DATA_LENGTH  # tr-te sınırını belirleyen yer


class Manager:
    """datamanipulatorun main kısmı"""

    def __init__(self):  # constructor
        self.data_training, self.data_test = self.prepare_data()

    def prepare_data(self):
        """Datayı oluşturuyor"""
        test = False
        data_tr, data_te = [], []
        data = pd.read_csv(DATA_DEST)
        data_ex = pd.read_csv(DATA_DEST_EX)

        for row_index in range(TOTAL_LENGTH - 1):
            target = data["Fark %"].iloc[row_index + 1]
            # onceki 5 gunu basar
            if row_index < 4:
                piece_now = list(data_ex["Fark %"].iloc[row_index + 11:])
                piece_now.extend(list(data["Fark %"].iloc[:row_index]))
            else:
                piece_now = list(data["Fark %"].iloc[row_index - 4: row_index])
            row_now = data.iloc[row_index]
            piece_now.append(row_now.iloc[6])
            piece_now.extend(list(row_now.iloc[8:]))
            piece_now.append(target)
            if test:
                if row_index % 10 == 9:
                    test = False
                elif row_index % 10 > row_index % 5:
                    continue
                else:
                    data_te.append(piece_now)
            else:
                if row_index % 10 == 9:
                    test = True
                elif row_index % 10 > row_index % 5:
                    continue
                else:
                    data_tr.append(piece_now)
        # print("tr: ", len(data_tr), len(data_te))
        return data_tr, data_te

    def plot(self):
        """Datayı çiziyor"""
        data = pd.read_csv(DATA_DEST)
        plt.plot(data.loc[:, "Fark %"])
        plt.show()

# Manager()

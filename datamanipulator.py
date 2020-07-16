"""csv ile uğraşılıp verinin alındığı modül"""
import pandas as pd
######################
####Constants#########
######################
DATA_DEST = "Data\\BIST_100_Gecmis_Verileri_Haftalik.csv"
TOTAL_LENGTH = pd.read_csv(DATA_DEST).shape[0]
TR_DATA_LENGTH, TE_DATA_LENGTH = 16, 8


class Manager:
    """datamanipulatorun main kısmı"""
    def __init__(self):  # constructor
        self.data_training, self.data_test = self.prepare_data()


    def prepare_data(self):
        """Datayı oluşturuyor"""
        data_tr, data_te = [], []
        bound = TR_DATA_LENGTH + TE_DATA_LENGTH - 2                         #tr-te sınırını belirleyen yer
        data = pd.read_csv(DATA_DEST)
        for row_index in range(TOTAL_LENGTH - 1):                           #sonuncu zaten kullanilamaz
            row_now = data.iloc[row_index]
            target = data["Fark"].iloc[row_index + 1]
            piece_now = list(row_now.iloc[8:])
            piece_now.append(target)
            if row_index % bound >= 0 and row_index % bound < 16:
                data_tr.append(piece_now)
            elif row_index % bound >= 16 and row_index % bound < bound:
                data_te.append(piece_now)
        return data_tr, data_te
    def plot(self):
        """Datayı çiziyor"""
        data = pd.read_csv(DATA_DEST)
        data.iloc[:, 7:].plot()

# Manager()

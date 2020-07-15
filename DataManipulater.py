import pandas as pd
import numpy as np
DATA_DEST = "BIST_100_Gecmis_Verileri_Haftalik.csv"
MODEL_DATA_DEST = "data_for_model.csv"
TOTAL_LENGTH = 0
PIECE_INDEX = 0
PIECE_LENGTH = 0

#dikkat! sadece data duzgun degilse kullan
##########################################
##csv dosyasını hizaya sokuyor
##########################################
class DataReorganizer:
    def reorganize(self, destination):
        start = True
        stri = ""
        header = ""
        piece = ""
        f = open(destination, "r")
        for line in f:
            if start:
                header = line.replace("\"", "")
                start = False
            else:
                line = line.replace("\"", "")
                line = line.replace("%", "")
                line = line.replace("B", "")
                line = line.replace("M", "")
                piece = ""
                for i in range(6):#virgulu bul, sonrasini degistir
                    index = line.find(",")
                    piece += line[:index+1]
                    line = line[index+1:]
                    index = line.find(".")
                    if index > 0:
                        line = line[:index]+line[index+1:]
                    index = line.find(",")
                    if index > 0:
                        line = line[:index]+"."+line[index+1:]
                    if i == 5:
                        piece += line
                #print(piece)
                stri = piece+stri
        stri = header+stri
        f.close()
        f = open(destination, "w")
        f.write(stri)

###################################
###X,Y ikililerini olusturuyor
###################################
class PredictorElaborator:
    def change_types(self):
        data = pd.read_csv(DATA_DEST).iloc[:, 1:]
        data = data.iloc[1:].astype(float)

    def add_diff(self):  # simdi-acilis yapmam lazim
        data = pd.read_csv(DATA_DEST)
        data["Fark"] = round(data["Simdi"]-data["Acilis"], 2)
        data = data.assign(Williams=self.add_williams_r())
        data = data.assign(MACD=self.add_macd())
        data = data.assign(Brent_Petrol=self.add_brent_petrol())
        data = data.assign(RSI=self.get_rsi())
        data = data.assign(USD_TR=self.add_usd())
        data = data[["Tarih", "Simdi", "Acilis", "Yuksek", "Dusuk", "Hac.", "Fark", "Fark %", "RSI", "Brent_Petrol", "USD_TR", "MACD", "Williams"]]
        data.to_csv(r'proje\\BIST_100_Gecmis_Verileri_Haftalik.csv', index=False, header=True)

    def get_rsi(self):
        data = pd.read_csv(DATA_DEST)
        avg_gain, avg_loss = 0, 0
        g_count, l_count = 0, 0
        rs = 0
        rsi = []
        for row_index in range(data.shape[0]):
            if(row_index != 0 and row_index%PIECE_LENGTH == 0):
                avg_gain = avg_gain/g_count
                avg_loss = avg_loss/l_count
                avg_loss = -avg_loss
                rs = avg_gain/avg_loss
                rsi.append(100-(100/(1+rs)))
                avg_gain, avg_loss = 0, 0
                g_count, l_count = 0, 0
            value = data["Fark %"].iloc[row_index]
            if value > 0:
                g_count += 1
                avg_gain += value
            else:
                l_count += 1
                avg_loss += value
        avg_gain = avg_gain/g_count
        avg_loss = avg_loss/l_count
        avg_loss = -avg_loss
        rs = avg_gain/avg_loss
        rsi.append(100-(100/(1+rs)))
        rsi_new = []
        for elem in rsi:
            for _ in range(25):
                rsi_new.append(round(elem, 2))
        return rsi_new[:data.shape[0]]

    def add_williams_r(self):
        data = pd.read_csv(DATA_DEST)
        arr_williams = []
        highest_high = data["Yuksek"].max()
        lowest_low = data["Dusuk"].min()
        for current_close in data["Simdi"]:
            current_close = (highest_high - current_close) / (highest_high - lowest_low)
            arr_williams.append(round(current_close * -100, 2))
        return arr_williams

    def add_usd(self):
        arr_usd = []
        usd = pd.read_csv("proje\\USD_TRY.csv")["Fark %"]
        for elem in usd:
            index = elem.index(',')
            elem = elem[:index]+'.'+elem[index+1:-1]
            arr_usd.insert(0, float(elem))
        return arr_usd

    def add_brent_petrol(self):
        arr_petrol = []
        petrol = pd.read_csv("proje\\Brent_Petrol.csv")["Fark %"]
        for elem in petrol:
            index = elem.index(',')
            elem = elem[:index]+'.'+elem[index+1:-1]
            arr_petrol.insert(0, float(elem))
        return arr_petrol

    def add_macd(self):
        arr_macd = []
        ema26, ema12 = 0, 0
        smoothing = 2
        data = pd.read_csv(DATA_DEST)
        ema12 = (data["Simdi"] * (smoothing / 13)) + (data["Acilis"] * (1 - (smoothing / 13)))
        ema26 = (data["Simdi"] * (smoothing / 27)) + (data["Acilis"] * (1 - (smoothing / 27)))
        print(ema26)
        arr_macd = round(ema12 - ema26, 2)
        return arr_macd

    def prepare_data(self):
        data = pd.read_csv(DATA_DEST)["Fark"]
        tr_next, tr_now = [], []
        te_next, te_now = [], []
        for i in enumerate(data):
            if i % 3 == 0:
                te_now.append(data[i])
            else:
                tr_now.append(data[i])
            if i > 0:
                if i%3 == 1:
                    te_next.append(data[i])
                else:
                    tr_next.append(data[i])
        if len(data) % 3 == 2:
            tr_next.append(0)
        else:
            te_next.append(0)
        print("len_tr:", len(tr_now), len(tr_next))
        print("len_te:", len(te_now), len(te_next))
        data_tr = {'tr_now':tr_now, 'tr_next':tr_next}
        data_te = {'te_now':te_now, 'te_next':te_next}
        data1 = pd.DataFrame(data_tr, columns=['tr_now', 'tr_next'])
        data2 = pd.DataFrame(data_te, columns=['te_now', 'te_next'])
        data_new = pd.concat([data1, data2], ignore_index=True, axis=1)
        data_new.columns = ['tr_now', 'tr_next', 'te_now', 'te_next' ]
        data_new.to_csv(r'data_for_model.csv', index=False, header=True)
    

#################################
#Ana class bu####################
#################################
class Manager:
    def __init__(self, length):  # constructor
        global PIECE_INDEX, TOTAL_LENGTH, PIECE_LENGTH
        #calculate total length
        TOTAL_LENGTH = len(pd.read_csv(MODEL_DATA_DEST)['tr_now'])
        #set length of pieces
        PIECE_LENGTH = length
        #index ayari
        PIECE_INDEX = -PIECE_LENGTH
        #####    
        #self.plot()

    def plot(self):
        data = pd.read_csv(DATA_DEST)
        
        data.iloc[:, 7:].plot()

    #parcalari buradan alacan da daha tam degil bu
    def get_training_piece(self):
        global PIECE_INDEX
        ########
        if PIECE_INDEX + PIECE_LENGTH > TOTAL_LENGTH:
            PIECE_INDEX = -PIECE_LENGTH
        ########
        PIECE_INDEX += PIECE_LENGTH
        data = np.array(pd.read_csv(MODEL_DATA_DEST).iloc[:, :2])
        if PIECE_INDEX + PIECE_LENGTH > TOTAL_LENGTH:
            data = data[PIECE_INDEX:TOTAL_LENGTH-1]
        else:
            data = data[PIECE_INDEX:PIECE_INDEX+PIECE_LENGTH]
        return data

    #sonunda mı diye kontrol ediyor
    #get_pieceyi çağırmadan kontrol etmeye kalkma
    def control_if_end(self):
        if PIECE_INDEX + PIECE_LENGTH > TOTAL_LENGTH:
            return True
        return False

    #kac parca varsa buradan alcan
    def get_number_of_pieces(self):
        return round(TOTAL_LENGTH / PIECE_LENGTH)

Manager(25)
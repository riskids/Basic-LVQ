import pandas as pd
import numpy as np

csv_dataset = pd.read_csv("dataset/datsetfull.csv", delimiter=',', header=0)
dataset = np.array(csv_dataset)                      #konversi dataset csv menjadi array
dataset = dataset.astype(float)                      #konversi dataset menjadi tipe float
n_dataset = len(dataset[:,0])                        #menghitung banyaknya dataset
n_fiturDataset = len(dataset[0,:]) - 1              #membaca jumlah fitur dataset 

csv_dataKlasifikasi = pd.read_csv("dataset/data_klasifikasi.csv", delimiter=',', header=0)
data_klasifikasi = np.array(csv_dataKlasifikasi)                #memasukan isi csv kedalam array
data_klasifikasiFloat = data_klasifikasi[:,1:5].astype(float)   #mengubah tipe data array ke float dan mengambil hanya colom index k 1-5
n_klasifikasi = len(data_klasifikasi[:,0])			            #menghitung banyaknya data klasifikasi
n_fiturKlasifikasi = len(data_klasifikasi[0,:]) - 1             #membaca jumlah fitur dataset 
dataFiture = dataset[:,:5]

datasetSiap = []
dataklasifikasiSiap = []


#menjadikan setiap baris pada array dataset menjadi array
for a in range(0, n_dataset):
    datasetSiap.append(dataFiture[a])

#menjadikan setiap baris pada array data_klasifikasi menjadi array
for a in range(0, n_klasifikasi):
    dataklasifikasiSiap.append(data_klasifikasiFloat[a])

# print("{:.2f}".format(1.123456789))
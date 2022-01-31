import pandas as pd
import numpy as np
import math
from dataset import datasetSiap, n_fiturDataset, n_fiturKlasifikasi, dataklasifikasiSiap, data_klasifikasi

csv_bobotAwal = pd.read_csv("dataset/bobot_modal.csv", delimiter=',', header=0)
bobotAwal = np.array(csv_bobotAwal)                         #konversi bobot csv menjadi array
iBobot = bobotAwal.astype(float)                               #konversi bobot menjadi tipe float

n_kelas = len(iBobot[:,0])                                   #menghitung banyaknya bobot
n_epoch = 5                                                  #Jumlah Epoh
lr = 0.05                                                    #learning Rate
penurunan = 0.01                                            #penurunan learning rate


def euclidean(x,y):
    jarak = 0.0
    for i in range (0,n_fiturDataset):
        jarak += (x[i]-y[i])**2  
    return float("{:.8f}".format(math.sqrt(jarak)))

def training(bobot,n_kelas,n_epoch,lr,penurunan):
    # Training
    itr = 0
    while(itr < n_epoch):
        alfa = lr                                                    
        reducingFactor = penurunan                                           
        
        # Menghitunga Banyak Data
        n_dataTraining	= len(datasetSiap)
        print("Training data berdasarkan dataset dan bobot yang dimasukan")
        print("Epoch ke - " + str(itr+1))
        for i in range(0, int(n_dataTraining)):
            data  =[]
            kelas =[]

            print("Iterasi Data ke - " + str(i+1))
            #Menghitung Jarak Data
            for j in range(0, n_kelas ):
                
                x = datasetSiap[i]
                y = bobot[j]
                # print("J = "+str(datasetSiap[i]))
                
                data.append(euclidean(x,y)) 
                kelas.append(bobot[j][n_fiturDataset])
                #print("bobot akhir = "+ str(kelas))
                dataJarak = {'jarak' : pd.Series(data),
                            'kelas' : pd.Series(kelas)} 
            # end for
            
            print("Cj = " +str(data))
            df = pd.DataFrame(dataJarak, columns = ['jarak', 'kelas'])
            df = df.sort_values(by=['jarak'])
            # print("df = "+str(df))
            
            kelasTemp = df.iloc[0,1]
            kelasTemp = int(kelasTemp)
            print("Cj minimum = W"+str(kelasTemp))

            if kelasTemp == datasetSiap[i][n_fiturDataset]:
                for x in range(0, n_fiturDataset):
                    bobot[kelasTemp-1][x] = bobot[kelasTemp-1][x] +  alfa * ( datasetSiap[i][x] - bobot[kelasTemp-1][x] ) 
                    # print("rumus (T=Cj): "+str(bobot[kelasTemp-1][x])+" + "+str(alfa)+" x ("+str(datasetSiap[i][x])+" - "+str(bobot[kelasTemp-1][x])+")")
            else:
                for x in range(0, n_fiturDataset):
                    bobot[kelasTemp-1][x] = bobot[kelasTemp-1][x] -  alfa * ( datasetSiap[i][x] - bobot[kelasTemp-1][x] )
                    # print("rumus (T><Cj): "+str(bobot[kelasTemp-1][x])+" - "+str(alfa)+" x ("+str(datasetSiap[i][x])+" - "+str(bobot[kelasTemp-1][x])+")")
            tampilBobotBaru = bobot[:,:4]
            print("W"+str(kelasTemp)+" baru = "+str(tampilBobotBaru[kelasTemp-1])+"\n")     
        itr+=1
        rF = reducingFactor * alfa
        alfa = alfa - rF
    tBobotakhir = pd.DataFrame(tampilBobotBaru, index=pd.Index(['W1','W2','W3']),columns=['','','',''])
    print("Diperoleh bobot akhir: "+str(tBobotakhir)+"\n")
   
    return bobot

def klasifikasi(bobot,dataklasifikasiSiap,n_fiturKlasifikasi,data_klasifikasi):
    print("Klasifikasi Data Berdasarkan Bobot Hasil Training!")
    n_dataKlasifikasi = len(dataklasifikasiSiap)
    keputusan = []
    ketentuan_keputusan = ['TUNDA','TIDAK','YA']
    for i in range(0, int(n_dataKlasifikasi)):
            data =[]
            kelas=[]

            print("Klasifikasi Data Ke - " + str(i+1))
            #Menghitung Jarak Data
            for j in range(0, n_kelas ):
                
                x = dataklasifikasiSiap[i]
                y = bobot[j]
                # print("J = "+str(datasetSiap[i]))
                
                data.append(euclidean(x,y)) 
                kelas.append(bobot[j][n_fiturKlasifikasi])
                #print("bobot akhir = "+ str(kelas))
                dataJarak = {'jarak' : pd.Series(data),
                            'kelas' : pd.Series(kelas)} 
            # end for
            
            print("Cj = " +str(data))
            df = pd.DataFrame(dataJarak, columns = ['jarak', 'kelas'])
            df = df.sort_values(by=['jarak'])
            # print("df = "+str(df))
            
            kelasTemp = df.iloc[0,1]
            kelasTemp = int(kelasTemp)
            keputusan += [ketentuan_keputusan[kelasTemp-1]]
            print("Jarak Terkecil adalah = W"+str(kelasTemp)+"\n")
    print("Berikut tabel klasifikasi hasil keputusan dengan ketentuan")
    print("W1 = Tunda")
    print("W2 = Tidak")
    print("W3 = Ya")
    keputusan = np.reshape(keputusan,(n_dataKlasifikasi,1))
    data_klasifikasi = np.append(data_klasifikasi,keputusan, axis=1)
    tData_klasifikasi = pd.DataFrame(data_klasifikasi, columns=['Nama UMKM','Lama Usaha','Jumlah Pekerjaan','Omzet','Jumlah Aset','Keputusan'])
    print(tData_klasifikasi)

#main program start here
bobotFinal = training(iBobot,n_kelas,n_epoch,lr,penurunan)
klasifikasi(bobotFinal,dataklasifikasiSiap,n_fiturKlasifikasi,data_klasifikasi)


import os
import numpy as np
from scipy.io import wavfile
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.fftpack import dct
import json

#wyswietlanie informacji o pliku
def informacje_o_pliku(dlugosc_pliku):
    print("Dlugosc pliku:", int(dlugosc_pliku),"s")


#wykres amplitudowy sygnalu
def probka_przed_analiza_wykres(audio, czas):
    plt.title('Probka przed analiza')
    plt.ylabel('Amplituda')
    plt.xlabel('Czas')
    plt.plot(czas, audio)
    plt.show()


#filtracja sygnalu filtrem dolnoprzepustowym
def preemfaza(audio):
    wspolczynnik_filtracji = 0.8
    audio_po_filtracji = np.append(audio[0], audio[1:] - wspolczynnik_filtracji * audio[:-1])

    return audio_po_filtracji


#wykres sygnalu po filtracji
def wykres_po_filtracji(audio, czas):
    plt.title('Preemfaza')
    plt.ylabel('Amplituda')
    plt.xlabel('Czas')
    plt.plot(czas, audio)
    plt.show()


#ramkowanie sygnalu
def ramkowanie_fft(audio_fft, czestotliwosc_probkowania, dlugosc_sygnalu, NFFT):
    rozmiar_ramki = 0.025 # 25ms
    rozmiar_kroku = 0.01  # 10ms

    dlugosc_ramki = rozmiar_ramki * czestotliwosc_probkowania
    dlugosc_ramki = int(round(dlugosc_ramki))
    krok = rozmiar_kroku * czestotliwosc_probkowania
    krok = int(round(krok))

    liczba_ramek = int(np.ceil(float(np.abs(dlugosc_sygnalu - dlugosc_ramki)) / krok))
    
    blok_ramek = liczba_ramek * krok + dlugosc_ramki
    z = np.zeros((blok_ramek - dlugosc_sygnalu))
    blok_sygnalu = np.append(audio_fft, z)

    wskazniki = np.tile(np.arange(0, dlugosc_ramki), (liczba_ramek, 1)) + np.tile(np.arange(0, liczba_ramek * krok, krok), (dlugosc_ramki, 1)).T
    ramki = blok_sygnalu[wskazniki.astype(np.int32, copy=False)]

    #okno Hamminga
    ramki *= np.hamming(dlugosc_ramki)

    #szybka transformata fouriera
    audio_fft = np.absolute(np.fft.rfft(ramki, NFFT))

    return audio_fft


def wykres_po_fft(audio, NFFT):
    periodogram = ((1.0 / NFFT) * ((audio) ** 2))\

    '''
    plt.title('Periodogram (wynik FFT)')
    plt.ylabel('Spektrum')
    plt.xlabel('Częstotliowść')
    plt.plot(periodogram)
    plt.show()
    '''
    
    return periodogram


def mele_na_hertze(mel):
    return 700*(10**(mel/2595.0)-1)


def hertze_na_mele(hz):
    # 2595 * numpy.log10(1 + hz/700.0)
    return 2595 * np.log10(1 + (hz/2) / 700.0)


#zestaw filtrow w skali mel
def filtry_mel(czestotliwosc_probkowania, NFFT, periodogram):
    liczba_filtrow = 40
    mel_nisko_czest = 0
    mel_wysoko_czest = hertze_na_mele(czestotliwosc_probkowania)
    punkty_mel = np.linspace(mel_nisko_czest, mel_wysoko_czest, liczba_filtrow + 2)
    punkty_hertz = mele_na_hertze(punkty_mel)
    bin = np.floor((NFFT + 1) * punkty_hertz / czestotliwosc_probkowania)

    z_filtrow = np.zeros((liczba_filtrow, int(np.floor(NFFT / 2 + 1))))

    for i in range(1, liczba_filtrow + 1):

        f_m_minus = int(bin[i - 1])
        f_m = int(bin[i])
        f_m_plus = int(bin[i + 1])

        for j in range(f_m_minus, f_m):\
            z_filtrow[i - 1, j] = (j - bin[i - 1]) / (bin[i] - bin[i - 1])

        for j in range(f_m, f_m_plus):
            z_filtrow[i - 1, j] = (bin[i + 1] - j) / (bin[i + 1] - bin[i])

    zespol_filtow = np.dot(periodogram, z_filtrow.T)
    zespol_filtow = np.where(zespol_filtow == 0, np.finfo(float).eps, zespol_filtow)
    zespol_filtow = 20 * np.log10(zespol_filtow)

    '''
    plt.title('Zestaw banków w skali Mel')
    plt.ylabel('Amplituda')
    plt.xlabel('Czestotliwosc')
    plt.plot(z_filtrow)
    plt.show()
    '''

    return zespol_filtow




def wykres_mfcc(mfcc, czas):
    plt.imshow(np.flipud(mfcc.T), cmap='jet', aspect=0.05, extent=[0,np.amax(czas),0,12])
    plt.title('Współczynniki MFCC')
    plt.ylabel('Liczba współczynnikow')
    plt.xlabel('Czas')
    plt.show()


#funkcja do analizy sygnału
def analiza_mfcc(SCIEZKA_PLIKU):

    #dane sygnalu
    czestotliwosc_probkowania, audio = wavfile.read(SCIEZKA_PLIKU)
    dlugosc_pliku = audio.shape[0] / czestotliwosc_probkowania
    czas = np.linspace(0, dlugosc_pliku, audio.shape[0])

    #probka przed analiza
    #informacje_o_pliku(dlugosc_pliku)
    '''probka_przed_analiza_wykres(audio, czas)'''

    #preemfaza
    audio = preemfaza(audio)
    dlugosc = audio.shape[0] / (2 * czestotliwosc_probkowania)
    czas = np.linspace(0, dlugosc, audio.shape[0])
    '''wykres_po_filtracji(audio, czas)'''

    #ramkowanie i fft
    NFFT = 512 # punkty nfft
    dlugosc = len(audio)
    audio = ramkowanie_fft(audio, czestotliwosc_probkowania, dlugosc, NFFT)
    periodogram = wykres_po_fft(audio, NFFT)

    #filtry mel
    zespol_filtow = filtry_mel(czestotliwosc_probkowania, NFFT, periodogram)
    
    #mfcc
    liczba_wspolczynnikow_ceps = 12
    mfcc = dct(zespol_filtow, type=2, axis=1, norm='ortho')[:, 1 : (liczba_wspolczynnikow_ceps + 1)]
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    '''wykres_mfcc(mfcc, czas)'''

    #konwersja na json
    lists = mfcc.tolist()
    json_mfcc = json.dumps(lists)

    return json_mfcc

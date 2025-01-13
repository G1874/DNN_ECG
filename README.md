### Zadanie:
Wykrywanie migotania przedsionków na sygnałach ECG.

### TODO
- [x] Klasa do trenowania sieci i walidacji
- [x] Dzielenie danych na 5-sekundowe fragmenty
- [x] Filtracja
- [x] Zbalansowanie ilości próbek danych klas
- [x] Dataset i Dataloader
- [x] Podzielić dataset na trening i walidacje
- [x] Transformacje (ToSpectrogram i ToTensor)
- [x] Pierwsza wersja sieci
- [x] Detekcja dostosowana do wymagań ewaluacji
- [ ] Sprawdzić działanie inferencji na testerce
- [ ] Zweryfikować poprawność kodu do kompozycji datasetu i treningu sieci

### Biblioteki:
* wfdb - pobieranie baz danych
* numpy 1.26.4 - starsza wersja żeby wfdb działało
* tqdm - terminal progress bar
* pandas
* torch

### Źródła:
1. Bazy danych - https://physionet.org/about/database/
    * https://physionet.org/content/afdb/1.0.0/
    * https://physionet.org/content/ltafdb/1.0.0/
2. Artykuły
    * [Detecting atrial fibrillation by deep convolutional neural networks](https://www-1webofscience-1com-1q5yy4oq600f4.wbg2.bg.agh.edu.pl/wos/woscc/full-record/WOS:000424187100009)

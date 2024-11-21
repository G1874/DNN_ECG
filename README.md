### Zadanie:
Wykrywanie migotania przedsionków na sygnałach ECG.

### TODO
- [x] Klasa do trenowania sieci i walidacji
- [x] Dzielenie danych na 5-sekundowe fragmenty
- [ ] Filtracja
- [ ] Zbalansowanie ilości próbek danych klas
- [ ] Dataset i Dataloader
- [ ] Pierwsza wersja sieci
- [ ] Detekcja dostosowana do wymagań Darka

### Biblioteki:
* wfdb - pobieranie baz danych
* numpy 1.26.4 - starsza wersja żeby wfdb działało

### Źródła:
1. Bazy danych - https://physionet.org/about/database/
    * https://physionet.org/content/afdb/1.0.0/
    * https://physionet.org/content/ltafdb/1.0.0/
2. Artykuły
    * [Detecting atrial fibrillation by deep convolutional neural networks](https://www-1webofscience-1com-1q5yy4oq600f4.wbg2.bg.agh.edu.pl/wos/woscc/full-record/WOS:000424187100009)

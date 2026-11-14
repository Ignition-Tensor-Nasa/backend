# NASA PSI — analiza i wizualizacje w Polars

Osobny moduł analityczny, uruchamiany z katalogu `backend`. Nie wymaga uruchamiania
FastAPI. Polars odpowiada za czyszczenie, konwersje jednostek, obliczenia, agregacje,
filtrowanie i redukcję liczby punktów. Plotly generuje interaktywne pliki HTML.
`openpyxl` służy wyłącznie do odczytu oryginalnych metadanych XLSX.

## Instalacja i uruchomienie

Wymagany Python 3.10 lub nowszy. Na macOS systemowe `python3` może być starsze —
wtedy użyj np. `python3.12` podczas tworzenia środowiska.

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r polars_analysis/requirements.txt
python -m polars_analysis
```

Instalacja przez główny `requirements.txt` również zawiera zależności analizy.
Otwórz `polars_analysis/outputs/index.html` w przeglądarce. Wszystkie wykresy
zawierają Plotly i po wygenerowaniu działają offline. Wybór z menu zmienia dane
wykresu; legenda pozwala ukrywać serie. Zoom na panelach CFI synchronizuje oś czasu.

Domyślne źródło to `../data` względem `backend`, wyznaczane na podstawie położenia
modułu, niezależnie od bieżącego katalogu. Można podać inny katalog:

```sh
python -m polars_analysis --data-dir /sciezka/do/data --output-dir /tmp/nasa-psi-analysis
python -m polars_analysis --test-id FLEX-1213 --max-points 800
python -m unittest polars_analysis.test_analysis -v
```

`--test-id` ogranicza tylko widok CFI; eksport i manifest nadal obejmują cały
wczytany zestaw. `--max-points` ogranicza każdą krzywą oddzielnie (minimum 4).
Przy ponownym uruchomieniu pliki wynikowe o tych samych nazwach są zastępowane.

## Źródła danych

W tej przestrzeni roboczej pliki NASA PSI zostały już pobrane do `data/` obok
repozytorium `backend`. Program pobiera z nich wybrany zestaw do
`outputs/sources/`, zachowując oryginalne nazwy i strukturę. Nie pobiera ponownie
archiwów z sieci i nie generuje danych syntetycznych.

Przy świeżym klonowaniu repozytorium dane należy uzyskać z
[katalogu NASA PSI](https://psi.nasa.gov/physci/repo). Nazwy badań można sprawdzić
w [oficjalnym wykazie NASA](https://www.nasa.gov/physical-sciences-informatics-psi/psi-investigations-by-research-area/).
Nie wpisano zgadywanych adresów pobierania ani DOI. Do publikacji należy uzupełnić
cytowania z „Cite this Investigation” przy odpowiednim badaniu w PSI.

Wymagane pliki wewnątrz katalogu przekazanego przez `--data-dir`:

```text
ACME-CFI-G_PSI-159/csv/
  PSI-159_Experimental table_CFI-G_Inverse Flames.csv
  PSI-159_Experimental table_CFI-G_Normal Flames.csv
CFI_PSI-39/txt/
  FLEX-*_DropletData.txt
  FLEX-*_CoolFlameData.txt
  FLEX-*_RadiometerData.txt
CFI_PSI-39/xlsx/
  CFI_DataSummary_Spreadsheet_Jan2021.xlsx
DAFT-DAFT-2_PSI-47/txt/
  DTrk*.txt
  PTrk*.txt
```

Nie każdy test CFI ma plik chłodnego płomienia — brak serii jest dopuszczalny.
XLSX jest odczytywany z arkusza `Master Sheet`, z zapisanymi wartościami komórek.
Program nie przelicza formuł Excela. Powtórzenie analizy z kopii użytych źródeł:

```sh
python -m polars_analysis --data-dir polars_analysis/outputs/sources --output-dir /tmp/psi-rerun
```

## Zgodność z VISUALIZATION_IDEAS.md

`INSTRUKCJA.md` był pusty podczas przygotowania tego modułu. Zakres wynika
z pomysłów oraz z weryfikacji rzeczywistych nagłówków i zawartości lokalnych plików.

| Pomysł | Implementacja i granice interpretacji |
| --- | --- |
| A. Exploration Atmosphere Flammability Heatmap | O₂ × ciśnienie; średnia tabelarycznej szybkości spalania lub Tad, osobno dla paliwa i Normal/Inverse. Brak automatycznego indeksu ryzyka i ekstrapolacji na habitaty. |
| B. Synchronized Droplet & Cool Flame Lifecycle Explorer | D², FWHM/P2P i dwie radiometrie; wybór testu, wspólny zoom, metadane O₂/He/paliwa/ciśnienia. Zachowane źródłowe czasy; synchronizacja fizyczna wymaga dokumentacji. |
| D. Smoke vs. Dust False-Alarm Signature Explorer | Część DAFT: P-Trak i DustTrak na osobnych panelach, z oryginalnymi jednostkami i początkami rejestracji. Bez klasyfikacji dym/pył i porównania SAME. |
| C, E, F, G i pozostała część D | Niezaimplementowane: wymagają dalszej weryfikacji kalibracji, geometrii, jednostek lub relacji metadanych. |

Weryfikacja pokazała m.in., że `SLICE_vel.csv` nie ma nagłówka, a
`CH(A)_1900K.csv` ma dwie kolumny bez oznaczenia grawitacji. Same nazwy tych plików
nie wystarczają do uczciwego wykresu przestrzennego lub porównania 0g/1g.
W SAFFIRE `RelativeSeconds` wymaga sprawdzenia jednostek: przyrosty około 1000
odpowiadają przyrostom znacznika czasu o około 1000 ms. Nie użyto tej kolumny
automatycznie jako sekund.

## Przetwarzanie i jakość

- ACME: usunięcie otaczających spacji w nagłówkach; usunięcie spacji z ID testu;
  konwersja udziału O₂ na procenty. `XO2` (Inverse) i `XO2 before test` (Normal)
  są mapowane do jednej kolumny. `HC burnrate (mg/s)` i `mHC (mg/s)` zachowują
  znaczenie źródłowej ilości tabelarycznej, bez nowej kalibracji.
- Wiersze bez ID testu są pomijane i zliczane. Niepoprawne warunki są oznaczane.
  Mapa uwzględnia wyłącznie `Hot ignition = yes` i dodatni czas spalania.
  `flash`, nieudane zapłony i błędne warunki pozostają w eksporcie, lecz nie na mapie.
- Mapa grupuje O₂ w przedziały po 5 punktów procentowych, a ciśnienie po 0,1 bar.
  Początek przedziału jest w tabeli Parquet, środek na wykresie. Hover pokazuje
  liczebność komórki. Nie ma interpolacji, wypełniania luk zerami ani mieszania paliw.
- Tad jest temperaturą adiabatyczną z tabeli, a nie pomiarem czujnika.
  Mapa średnich dla zapalonych testów nie wyznacza prawdopodobieństwa pożaru.
- CFI: format długi, źródłowy czas w sekundach, dodatkowo D² w mm².
  Weryfikowana jest faktyczna zawartość plików: w lokalnym archiwum
  `FLEX-1227_RadiometerData.txt` i `FLEX-1239_RadiometerData.txt` zawierają PDF,
  a `FLEX-1248_DropletData.txt` i `FLEX-1248_RadiometerData.txt` mają zawartość
  binarną zamiast TSV. Są pomijane, wykazywane w `excluded_files` w raporcie
  i zachowywane w kopii źródeł. Nie zamienia się ich na sztuczne serie.
  Wartości ujemne oraz niefinitywne są oznaczane, a źródła zachowane.
  W radiometrii występują duże wartości ujemne; widok pokazuje je bez korekty,
  na osobnych osiach dla szerokiego i wąskiego pasma. Wymagają sprawdzenia kalibracji.
  Nie dopasowano automatycznie stałej spalania K bez wyboru fazy spalania.
- DAFT: nagłówek instrumentu i liczba próbek są sprawdzane.
  Plik `DTrk0002.txt` ma 779 próbek przy deklarowanych 778: dodatkowa próbka
  jest zachowana i oznaczona `beyond_declared_count`. Końcowe bajty NUL
  w `PTrk0003.txt` są pomijane jako dopełnienie pliku. Ucięta seria powoduje błąd.
  Raport zawiera deklarowane i rzeczywiste liczebności dla każdego źródła.
  Brak potwierdzonego
  interwału pomiarowego oznacza oś numeru próbki, nie czasu. Nie łączy się serii
  po samym TestID: takie same numery dotyczą rejestracji o różnych początkach.
- Redukcja punktów wykresów używa obwiedni minimum/maksimum w kolejnych grupach
  próbek, zachowuje ekstrema i końce krzywej. To metoda min/max, nie LTTB.
  Pełne dane pozostają w Parquet; limit wykresu nie zmienia agregacji naukowych.

## Wyniki

W `outputs/` powstają trzy wykresy HTML, strona `index.html`, `report.json`
z liczebnościami, jakością danych i manifestem SHA-256 oraz:

| Plik w processed/ | Zawartość |
| --- | --- |
| acme_conditions.parquet i .csv | Znormalizowane warunki testów, statusy, źródła |
| acme_burn_rate_grid.parquet | Średnie szybkości spalania i liczebności komórek |
| acme_tad_grid.parquet | Średnie Tad i liczebności komórek |
| cfi_metadata.parquet i .csv | Metadane z Master Sheet, powiązane przez FLEX ID |
| cfi_series.parquet | Pełne serie i flagi jakości, łącznie z D² |
| daft_series.parquet | Próbki, instrumenty, jednostki i oryginalne początki rejestracji |

Źródła i wyniki są ignorowane przez Git. Kod nie wykonuje operacji Git.

## Walidacja w środowisku przygotowania

Sprawdzono składnię Pythona oraz formaty i nagłówki lokalnych plików źródłowych.
Dołączone testy dotyczą separacji paliw i odrzucania nieudanych zapłonów,
obliczania D², flagowania ujemnej radiometrii, zachowania ekstremów przy redukcji
oraz odrzucania uciętych plików DAFT.

W środowisku przygotowania brakowało Polars i Plotly, a próba instalacji pip
zakończyła się błędem dostępu sieciowego. Pełne uruchomienie, testy zależne od
tych bibliotek oraz wizualna kontrola HTML pozostają do wykonania po instalacji.

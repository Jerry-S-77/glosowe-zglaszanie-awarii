# Głosowe zgłaszanie awarii

Projekt własny z kursu „Od zera do AI” (moduł 8).

Pracownicy utrzymania ruchu przy awarii mają zajęte ręce, pracują w hałasie i w rękawicach - zamiast
wypełniać formularz, technik nagrywa krótki opis głosem. Aplikacja zamienia nagranie na uporządkowane zgłoszenie, a po zatwierdzeniu
pokazuje podobne awarie z przeszłości razem z przyczyną i tym, co wtedy pomogło.

## Jak to działa

1. **Opis awarii** - nagrywam opis w przeglądarce albo wgrywam plik audio (mp3, wav, m4a, ogg).
   Opcjonalnie dodaję zdjęcie tabliczki znamionowej albo ekranu HMI.
2. **Transkrypcja** - `whisper-1` zamienia nagranie na tekst. Tekst mogę poprawić, zanim pójdzie dalej.
3. **Zgłoszenie** - `gpt-4o` z biblioteką `instructor` układa z transkrypcji (i danych ze zdjęcia) zgłoszenie:
   system, urządzenie, lokalizacja, objaw, kod alarmu, priorytet, wpływ na bezpieczeństwo, wpływ na GMP,
   działanie natychmiastowe i braki informacji. Pola zamknięte (system, priorytet, wpływy) to listy
   `Literal`, więc model musi wybrać jedną z dozwolonych wartości.
4. **Zatwierdzenie** - zgłoszenie poprawiam w formularzu i zatwierdzam. Dopiero wtedy liczę embedding
   (`text-embedding-3-large`) i zapisuję zgłoszenie w Qdrant. Zgłoszenie mogę pobrać jako JSON albo CSV.
5. **Podobne awarie** - aplikacja pokazuje 5 najbardziej podobnych awarii z historii: objaw, przyczynę,
   co pomogło i czas przestoju.

Dodatkowo:

* **Zdjęcie tabliczki / HMI** - `gpt-4o` odczytuje typ, numer seryjny, kod alarmu i komunikat. Kod alarmu
  ze zdjęcia ma pierwszeństwo przed podyktowanym, a rozbieżność trafia do braków informacji.
* **Koszt zgłoszenia** - w panelu bocznym widzę koszt każdego etapu (tokeny, minuty nagrania) w USD i PLN.
  Zgłoszenie ze zdjęciem kosztuje około 0,01 USD (ok. 4 grosze), bez zdjęcia około 0,005 USD
  (ceny OpenAI i kurs NBP z października 2026).
* **Szukaj w historii** - mogę opisać awarię własnymi słowami i od razu zobaczyć podobne przypadki.
* **Historia zgłoszeń** - lista zatwierdzonych zgłoszeń z filtrami (system, priorytet, wpływ na GMP, tekst,
  daty) i eksportem do CSV.

Priorytet oraz wpływ na bezpieczeństwo i GMP to tylko propozycja modelu. Aplikacja jest narzędziem
pomocniczym - nie zastępuje procedur zakładowych ani decyzji osoby odpowiedzialnej i nie jest systemem
zwalidowanym w rozumieniu GxP.

## Pliki

| Plik / folder | Co zawiera |
|---|---|
| `app.py` | aplikacja Streamlit |
| `zgloszenie.py` | wspólna struktura zgłoszenia (Pydantic) - używa jej aplikacja i notebook |
| `historia_awarii.ipynb` | generowanie, przegląd, redakcja i import syntetycznej historii awarii |
| `data/historia_awarii_surowa.json` | historia wygenerowana przez `gpt-4o-mini` (przed redakcją) |
| `data/redakcja_historii.json` | moje poprawki do każdego rekordu |
| `data/historia_awarii.json` | historia po redakcji - to ją ładuję do Qdrant |
| `przyklady/` | nagrania i zdjęcia do wypróbowania aplikacji |
| `requirements.txt`, `packages.txt` | biblioteki Pythona i `ffmpeg` do wdrożenia |
| `.env.example` | wzór pliku `.env` |

## Uruchomienie u siebie

1. Środowisko z Pythonem 3.11 i bibliotekami:

   ```
   conda create -n glosowe_awarie python=3.11
   conda activate glosowe_awarie
   conda install -c conda-forge ffmpeg
   pip install -r requirements.txt
   ```

2. Plik `.env` na wzór `.env.example`:
   * `OPENAI_API_KEY` - klucz OpenAI (gdy go nie ma, aplikacja poprosi o klucz na stronie),
   * `QDRANT_URL` i `QDRANT_API_KEY` - dane klastra Qdrant Cloud. Gdy ich nie ma, aplikacja zakłada lokalną
     bazę w folderze `qdrant_db` - wtedy historię trzeba najpierw zaimportować notebookiem.

3. Historia awarii: w notebooku `historia_awarii.ipynb` wystarczy uruchomić wszystkie komórki. Surowa
   historia jest już w repozytorium, więc notebook jej nie generuje drugi raz - składa wersję po redakcji
   i ładuje ją do Qdrant (koszt embeddingów 150 rekordów to ułamek centa).

4. Start aplikacji:

   ```
   streamlit run app.py
   ```

## Wdrożenie (Streamlit Community Cloud)

Tak jak „Audio Notatki” z modułu 7:

* repozytorium z tymi plikami (bez `.env` - pilnuje tego `.gitignore`),
* w ustawieniach aplikacji wersja Pythona 3.11,
* w **Secrets** dane do Qdrant Cloud:

  ```
  QDRANT_URL = "https://...cloud.qdrant.io:6333"
  QDRANT_API_KEY = "..."
  ```

* klucza OpenAI nie wpisuję w sekrety - każdy użytkownik podaje własny na stronie.

Darmowy klaster Qdrant Cloud usypia się po tygodniu bez ruchu, a po 4 tygodniach uśpienia jest usuwany.
Uśpiony klaster aplikacja zgłasza komunikatem, że nie może połączyć się z bazą - trzeba zalogować się do
Qdrant Cloud i wznowić klaster. Gdyby klaster został usunięty, wystarczy założyć nowy, podmienić sekrety
i uruchomić notebook `historia_awarii.ipynb` - historia odtworzy się z plików w `data/`.

## Czym różni się od kursu

Bazą był kod „Audio Notatek” (moduł 7) i lekcje o modelach z modułu 8. Zmiany:

* **Dane syntetyczne** - nie ma publicznego zbioru awarii infrastruktury farmaceutycznej, a prawdziwych
  zgłoszeń awarii nie można publikować. Historię 150 awarii wygenerowałem `gpt-4o-mini`, ale po
  przeczytaniu okazała się tylko szkieletem (błędy merytoryczne, powtarzające się kody alarmów, mylone
  lokalizacje). Wszystkie rekordy poprawiłem - poprawki są w `data/redakcja_historii.json`,
  a notebook pokazuje, co i ile razy zmieniłem.
* **Qdrant Cloud** zamiast bazy w pamięci - dane sekretów biorę z `st.secrets`, jak przy wdrażaniu w module 7.
  Lokalnie pliku z sekretami nie mam, a Streamlit rzuca wtedy błędem `StreamlitSecretNotFoundError`, więc go
  łapię.
* **Wgrywanie pliku audio** obok nagrywania i `language="pl"` w transkrypcji.
* **Opisy pól** (`Field(description=...)`) w strukturze zgłoszenia i osobny plik `zgloszenie.py`, bo tę samą
  strukturę używa notebook.
* **Generowanie historii** w trybie `instructor.Mode.TOOLS_STRICT` z `temperature=1` - bez tego
  `gpt-4o-mini` gubił pola w liście awarii.
* **Numery w bazie** - historia ma numery od 1 000 000, żeby nie mieszała się ze zgłoszeniami z aplikacji.
  Wyszukiwanie podobnych awarii filtruje po polu `zrodlo` (indeks w Qdrant), bo tylko historia ma przyczynę
  i działanie naprawcze.
* **Zdjęcie** - polecenie dla modelu ma reguły z testów: oznaczenie urządzenia to nie lokalizacja, kod ze
  zdjęcia wygrywa z podyktowanym, brak zdjęcia nie jest brakiem informacji.
* **Koszty** - wzór z „NaszGPT” (moduł 6, v4), ale z aktualnym cennikiem `gpt-4o`, ceną `whisper-1` za minutę
  i embeddingów. Tokeny z `instructor` biorę przez `create_with_completion`.
* **Zły klucz OpenAI** - komunikat jak w moim zadaniu domowym z modułu 9, ale klucz sprawdzam raz po
  wpisaniu (`models.list()`), bo `instructor` opakowuje błąd złego klucza w swój wyjątek.
* **Wersje bibliotek** w `requirements.txt` są przypięte do tych, na których testowałem.

## Jak testowałem

Każdą wersję (od nagrywania do historii zgłoszeń) sprawdzałem w przeglądarce, a na koniec przeszedłem całość
od nowa po świeżym uruchomieniu. Nagrania w `przyklady/` zrobiłem syntezatorem mowy, zdjęcia wygenerowałem
(tabliczka ma odblask, szum, perspektywę i uciętą krawędź, jak zdjęcie z telefonu).

| Scenariusz | Pliki | Wynik |
|---|---|---|
| Rozbieżny kod alarmu | `awaria_ahu03.wav` + `ekran_hmi_ahu03.png` | technik mówi A112, na ekranie A121 - w zgłoszeniu A121, rozbieżność w brakach informacji; po zatwierdzeniu 5 podobnych awarii HVAC |
| Niepełny opis | `pompa_obiegowa_bez_lokalizacji.wav` | lokalizacja i kod alarmu puste i dopisane do braków; brak zdjęcia nie jest brakiem |
| Zdjęcie tabliczki | `silnik_wentylatora_w12.wav` + `tabliczka_silnika_w12.jpg` | odczytany typ, numer fabryczny i oznaczenie W-12; W-12 trafia do urządzenia, a nie do lokalizacji |

Sprawdziłem też: poprawkę transkrypcji i pól formularza przed zatwierdzeniem, ponowne zatwierdzenie
(nadpisuje to samo zgłoszenie zamiast tworzyć duplikat), czyszczenie kolejnych kroków po zmianie nagrania
lub zdjęcia, pobieranie JSON i CSV (polskie znaki w Excelu), filtry i eksport historii zgłoszeń, aplikację
bez `.env` (prośba o klucz) i z nieprawidłowym kluczem (komunikat i ponowne pole).

W bazie zostawiłem kilka zgłoszeń z testów (numery poniżej miliona), żeby zakładka „Historia zgłoszeń”
nie była pusta. Nie biorą udziału w szukaniu podobnych awarii.

Co zauważyłem przy testach:

* podobieństwo w wynikach mieści się zwykle w wąskim zakresie (ok. 0,6-0,85), bo wszystkie teksty mają
  ten sam układ (system, urządzenie, objaw, kod) - kolejność wyników jest jednak sensowna,
* model przy nieczytelnym fragmencie zdjęcia potrafi coś dopowiedzieć (np. uciętą literę) - dlatego dane ze
  zdjęcia zawsze widać przed utworzeniem zgłoszenia.

## Co dalej

* zamykanie zgłoszeń: dopisanie przyczyny i działania naprawczego, żeby zgłoszenia z aplikacji same stawały
  się historią i trafiały do podobnych awarii,
* prawdziwa historia z systemu utrzymania ruchu zamiast danych syntetycznych (po anonimizacji),
* logowanie użytkowników i przypisanie zgłoszenia do technika.

# Wspólna struktura zgłoszenia. Trzymam ją w osobnym pliku, bo korzysta z niej
# i aplikacja (app.py), i notebook z historią awarii (historia_awarii.ipynb).
from typing import Literal

from pydantic import BaseModel, Field

# Listy zamknięte - model musi wybrać jedną z tych wartości (Literal w Pydantic),
# a ja mam potem porządek w filtrach i w bazie
SYSTEMY = Literal[
    "HVAC",
    "Woda oczyszczona",
    "Izolatory",
    "Automatyka",
    "Sprężone powietrze i gazy",
    "Zasilanie",
    "Inne",
]
PRIORYTETY = Literal["niski", "średni", "wysoki", "krytyczny"]
WPLYWY = Literal["brak", "możliwy", "tak"]


# Struktura zgłoszenia - tak jak klasy Person / ShortStory w lekcji text-to-json
# (moduł 8, lekcja 15). Opisy pól dopisałem, żeby model wiedział, co gdzie wpisać.
class Zgloszenie(BaseModel):
    system: SYSTEMY = Field(description="System, którego dotyczy awaria")
    urzadzenie: str = Field(description="Nazwa lub numer urządzenia, np. 'centrala wentylacyjna nr 3'")
    lokalizacja: str = Field(description="Gdzie jest urządzenie: budynek, piętro, pomieszczenie")
    objaw: str = Field(description="Co się dzieje, 1-2 zdania")
    kod_alarmu: str = Field(description="Kod alarmu z panelu lub HMI, jeśli technik go podał")
    priorytet: PRIORYTETY = Field(description="Proponowany priorytet zgłoszenia")
    wplyw_na_bezpieczenstwo: WPLYWY = Field(description="Czy awaria zagraża ludziom")
    wplyw_na_gmp: WPLYWY = Field(description="Czy awaria może wpłynąć na jakość produktu lub warunki GMP")
    dzialanie_natychmiastowe: str = Field(description="Co technik już zrobił na miejscu")
    braki_informacji: list[str] = Field(description="Czego brakuje w opisie, żeby zgłoszenie było kompletne")


# To, co da się odczytać ze zdjęcia tabliczki znamionowej albo ekranu HMI.
# Wzór jak klasa ImageInfo z lekcji image-to-json (moduł 8, lekcja 20).
class DaneZeZdjecia(BaseModel):
    rodzaj_zdjecia: Literal["tabliczka znamionowa", "ekran HMI", "inne"] = Field(
        description="Co jest na zdjęciu"
    )
    urzadzenie: str = Field(description="Nazwa, typ lub oznaczenie urządzenia widoczne na zdjęciu")
    numer_seryjny: str = Field(description="Numer seryjny lub fabryczny, jeśli jest widoczny")
    kod_alarmu: str = Field(description="Kod alarmu lub błędu widoczny na ekranie")
    komunikat: str = Field(description="Treść alarmu lub komunikatu z ekranu, przepisana dosłownie")
    inne_odczyty: list[str] = Field(
        description="Inne czytelne dane, np. parametry z wartościami i jednostkami"
    )


# Awaria z historii to zgłoszenie plus to, co wiadomo dopiero po naprawie
class AwariaHistoryczna(Zgloszenie):
    przyczyna: str = Field(description="Ustalona przyczyna awarii, 1-2 zdania")
    dzialanie_naprawcze: str = Field(description="Co naprawiono i jak, 1-2 zdania")
    czas_przestoju_h: float = Field(description="Czas przestoju w godzinach")


# Do embeddingu biorę tylko to, co opisuje samą awarię. Priorytet czy lokalizacja
# nie mówią, czy dwie awarie są do siebie podobne.
def ticket_to_text(zgloszenie):
    return (
        f'System: {zgloszenie["system"]}. '
        f'Urządzenie: {zgloszenie["urzadzenie"]}. '
        f'Objaw: {zgloszenie["objaw"]} '
        f'Kod alarmu: {zgloszenie["kod_alarmu"]}'
    )

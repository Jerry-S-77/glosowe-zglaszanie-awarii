import base64
import json
from datetime import datetime
from hashlib import md5
from io import BytesIO
from typing import get_args

import instructor
import pandas as pd
import streamlit as st
from audiorecorder import audiorecorder  # type: ignore
from dotenv import dotenv_values
from openai import AuthenticationError, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from streamlit.errors import StreamlitSecretNotFoundError

# Strukturę zgłoszenia mam w osobnym pliku, bo używa jej też notebook z historią awarii
from zgloszenie import (
    PRIORYTETY,
    SYSTEMY,
    WPLYWY,
    DaneZeZdjecia,
    Zgloszenie,
    ticket_to_text,
)

# Lokalnie klucz OpenAI trzymam w pliku .env (tak jak w całym kursie).
# Po wdrożeniu pliku .env nie ma - wtedy aplikacja poprosi o klucz na stronie.
env = dotenv_values(".env")

# Dane do Qdrant Cloud biorę z sekretów Streamlita, tak jak przy wdrażaniu
# "Audio Notatek" (moduł 7, lekcja 25). Lokalnie pliku z sekretami nie mam,
# a nowy Streamlit rzuca wtedy błędem, więc go łapię.
try:
    if "QDRANT_URL" in st.secrets:
        env["QDRANT_URL"] = st.secrets["QDRANT_URL"]
    if "QDRANT_API_KEY" in st.secrets:
        env["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
except StreamlitSecretNotFoundError:
    pass

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

TICKET_MODEL = "gpt-4o"

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

QDRANT_COLLECTION_NAME = "awarie"

# Lokalnie baza siedzi w tym folderze (w kursie była w pamięci i znikała po restarcie)
QDRANT_LOCAL_PATH = "qdrant_db"

# Cennik jak model_pricings w "NaszGPT liczy koszty" (moduł 6, wersja v4).
# Ceny w USD z cennika OpenAI (sprawdzone 24.10.2026) - gpt-4o od kursu potaniał.
# whisper-1 liczy się za minutę nagrania, embeddingi tylko za tokeny wejściowe.
model_pricings = {
    "gpt-4o": {
        "input_tokens": 2.50 / 1_000_000,  # za token
        "output_tokens": 10.00 / 1_000_000,  # za token
    },
    "text-embedding-3-large": {
        "input_tokens": 0.13 / 1_000_000,  # za token
    },
    "whisper-1": {
        "minute": 0.006,  # za minutę nagrania
    },
}
# Średni kurs NBP z 23.10.2026 - do zmiany, gdy mocno się rozjedzie
USD_TO_PLN = 3.92


# Każde wywołanie OpenAI dopisuję do listy kosztów bieżącego zgłoszenia
# i do sumy całej sesji (wyszukiwanie w historii nie należy do żadnego zgłoszenia)
def zapisz_koszt(etap, model, zuzycie, koszt_usd, do_zgloszenia=True):
    if do_zgloszenia:
        st.session_state["koszty_zgloszenia"].append({
            "etap": etap,
            "model": model,
            "zużycie": zuzycie,
            "koszt_usd": koszt_usd,
        })
    st.session_state["koszt_sesji_usd"] += koszt_usd
    pokaz_koszty()


def koszt_czatu(etap, completion):
    pricing = model_pricings[TICKET_MODEL]
    usage = completion.usage
    koszt = (
        usage.prompt_tokens * pricing["input_tokens"]
        + usage.completion_tokens * pricing["output_tokens"]
    )
    zapisz_koszt(
        etap,
        TICKET_MODEL,
        f"{usage.prompt_tokens} tok. wej. + {usage.completion_tokens} tok. wyj.",
        koszt,
    )


def koszt_zgloszenia_usd():
    return sum(k["koszt_usd"] for k in st.session_state["koszty_zgloszenia"])


# Panel boczny z kosztami - metryki USD i PLN jak w module 6.
# Rysuję go w placeholderze, bo koszty dochodzą w trakcie przebiegu skryptu,
# a dalej w kodzie są st.stop() i panel na końcu by się nie pokazał.
def pokaz_koszty():
    with koszty_placeholder.container():
        st.subheader("Koszt zgłoszenia")
        koszt = koszt_zgloszenia_usd()
        c0, c1 = st.columns(2)
        with c0:
            st.metric("USD", f"${koszt:.4f}")
        with c1:
            st.metric("PLN", f"{koszt * USD_TO_PLN:.4f}")

        # Tabela w wąskim panelu ucinała kolumny, więc każdy etap to osobna linijka.
        # Wszystkie linijki daję w jednym st.markdown: panel przerysowuję kilka razy w jednym
        # przebiegu i przy osobnych elementach krótsza lista nie zakrywała dłuższej.
        st.markdown("  \n".join(
            f':small[**{k["etap"]}** ({k["model"]}): {k["zużycie"]} → ${k["koszt_usd"]:.5f}]'
            for k in st.session_state["koszty_zgloszenia"]
        ))

        st.caption(
            f'Cała sesja (razem z wyszukiwaniem): ${st.session_state["koszt_sesji_usd"]:.4f} '
            f'/ {st.session_state["koszt_sesji_usd"] * USD_TO_PLN:.4f} PLN'
        )


# Klienta OpenAI tworzę z klucza zapisanego w session_state,
# wzór z aplikacji "Audio Notatki" (moduł 7)
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


# Transkrypcja jak w "Audio Notatkach" i lekcji o whisper-1 (moduł 8, lekcja 21)
def transcribe_audio(audio_bytes, audio_nazwa):
    openai_client = get_openai_client()
    # OpenAI rozpoznaje format po nazwie pliku, więc muszę ją nadać.
    # W kursie było zawsze "audio.mp3", u mnie może to być też wgrany wav albo m4a.
    audio_file = BytesIO(audio_bytes)
    audio_file.name = audio_nazwa
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        # Tego w kursie nie było: zgłoszenia są zawsze po polsku,
        # a podpowiedź języka pomaga przy krótkich nagraniach.
        language="pl",
        response_format="verbose_json",
    )

    # verbose_json podaje długość nagrania w sekundach - za to płacę przy whisper-1
    minuty = transcript.duration / 60
    zapisz_koszt(
        "transkrypcja",
        AUDIO_TRANSCRIBE_MODEL,
        f"{minuty:.2f} min nagrania",
        minuty * model_pricings[AUDIO_TRANSCRIBE_MODEL]["minute"],
    )

    return transcript.text


# Zdjęcie dla OpenAI musi być tekstem base64 - jak prepare_image_for_open_ai
# z lekcji o obrazach (moduł 8, lekcje 19-20), tylko u mnie z bajtów, a nie z pliku
def prepare_image_for_open_ai(image_bytes, image_type):
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    return f"data:{image_type};base64,{image_data}"


# Zdjęcie -> JSON, wzór z generate_data_for_image (moduł 8, lekcja 20)
def read_image_data(image_bytes, image_type):
    instructor_openai_client = instructor.from_openai(get_openai_client())
    # create_with_completion zamiast create, bo oprócz danych potrzebuję usage
    # z odpowiedzi OpenAI (tokeny do liczenia kosztu)
    res, completion = instructor_openai_client.chat.completions.create_with_completion(
        model=TICKET_MODEL,
        temperature=0,
        response_model=DaneZeZdjecia,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "To zdjęcie tabliczki znamionowej urządzenia albo ekranu HMI. "
                            "Przepisz tylko to, co jest czytelne. Jeśli czegoś nie widać, zostaw pole puste, "
                            "niczego nie zgaduj."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_open_ai(image_bytes, image_type),
                            "detail": "high",
                        },
                    },
                ],
            },
        ],
    )
    koszt_czatu("odczyt zdjęcia", completion)

    return res.model_dump()


# Tekst -> JSON przez instructor, wzór z funkcji retrieve_structure (moduł 8, lekcja 15)
def create_ticket(transkrypcja, dane_ze_zdjecia=None):
    # Dane ze zdjęcia doklejam do polecenia tylko wtedy, gdy technik dodał zdjęcie
    if dane_ze_zdjecia:
        zdjecie_tekst = (
            "Dane odczytane ze zdjęcia tabliczki znamionowej lub ekranu HMI:\n"
            + json.dumps(dane_ze_zdjecia, ensure_ascii=False, indent=2)
        )
    else:
        # Zdjęcie jest opcjonalne. Gdy pisałem tu "brak zdjęcia", model dopisywał to do braków informacji.
        zdjecie_tekst = ""

    instructor_openai_client = instructor.from_openai(get_openai_client())
    res, completion = instructor_openai_client.chat.completions.create_with_completion(
        model=TICKET_MODEL,
        temperature=0,
        response_model=Zgloszenie,
        messages=[
            {
                "role": "user",
                "content": f"""
Pracujesz w dziale infrastruktury technicznej zakładu farmaceutycznego (HVAC, woda oczyszczona,
izolatory, automatyka). Z opisu awarii podanego głosem przez technika przygotuj zgłoszenie.

Zasady:
- korzystaj tylko z tego, co powiedział technik, i z danych odczytanych ze zdjęcia (jeśli je dodał), niczego nie dopowiadaj,
- zdjęcie jest opcjonalne, jego brak nie jest brakiem informacji,
- kod alarmu bierz ze zdjęcia, jeśli jest tam czytelny - technik mógł go przekręcić,
- urządzenie opisz tak, żeby dało się je znaleźć: do czego służy (z opisu technika) plus oznaczenie
  i typ ze zdjęcia, np. "silnik wentylatora wyciągowego W-12, typ SKg 132M-4B",
- oznaczenie urządzenia (np. W-12, AHU-03) to nie jest lokalizacja; lokalizacja to budynek, piętro, pomieszczenie,
- jeśli technik podał inny kod alarmu albo inne urządzenie niż widać na zdjęciu, wpisz wartość ze zdjęcia,
  a rozbieżność opisz w braki_informacji, żeby człowiek to sprawdził,
- jeśli jakiejś informacji potrzebnej w zgłoszeniu nie ma ani w opisie, ani na zdjęciu, zostaw pole tekstowe puste
  i dopisz ten brak do braki_informacji,
- puste pola w danych ze zdjęcia (np. brak numeru seryjnego) nie są brakami w zgłoszeniu,
- priorytet oraz wpływ na bezpieczeństwo i GMP to tylko propozycja, ostatecznie zatwierdza je człowiek,
- pisz po polsku, krótko i rzeczowo.

Opis awarii:
{transkrypcja}

{zdjecie_tekst}
                """,
            },
        ],
    )
    koszt_czatu("zgłoszenie", completion)

    return res.model_dump()


#
# DB
#
# Cała ta część to przeróbka bazy z "Audio Notatek" (moduł 7, lekcje 11-13 i 24)
@st.cache_resource
def get_qdrant_client():
    # Są dane do chmury -> Qdrant Cloud (potrzebne po wdrożeniu).
    # Nie ma -> baza w folderze na dysku, do pracy u siebie.
    if env.get("QDRANT_URL") and env.get("QDRANT_API_KEY"):
        return QdrantClient(
            url=env["QDRANT_URL"],
            api_key=env["QDRANT_API_KEY"],
        )

    return QdrantClient(path=QDRANT_LOCAL_PATH)


# cache_resource, żeby sprawdzać bazę raz na uruchomienie aplikacji, a nie przy każdym kliknięciu
@st.cache_resource
def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )

    # Indeks po polu "zrodlo" - Qdrant Cloud potrzebuje go, żeby filtrować
    # wyszukiwanie tylko do awarii z historii (tego w kursie nie było)
    qdrant_client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="zrodlo",
        field_schema="keyword",
    )


def get_embedding(text, etap, do_zgloszenia=True):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    tokeny = result.usage.prompt_tokens
    zapisz_koszt(
        etap,
        EMBEDDING_MODEL,
        f"{tokeny} tok.",
        tokeny * model_pricings[EMBEDDING_MODEL]["input_tokens"],
        do_zgloszenia=do_zgloszenia,
    )

    return result.data[0].embedding


def count_tickets_in_db():
    qdrant_client = get_qdrant_client()
    return qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    ).count


def add_ticket_to_db(zgloszenie, point_id=None):
    qdrant_client = get_qdrant_client()
    # Nowe zgłoszenie dostaje kolejny numer, jak notatki w kursie.
    # Gdy poprawiam już zapisane zgłoszenie, podaję jego numer i nadpisuję punkt,
    # żeby w bazie nie zrobił się duplikat.
    if point_id is None:
        point_id = count_tickets_in_db() + 1

    # Embedding liczę przed zapisem, żeby jego koszt wszedł już do kosztu zapisanego w bazie
    vector = get_embedding(text=ticket_to_text(zgloszenie), etap="zapis w bazie")

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    **zgloszenie,
                    "zrodlo": "aplikacja",
                    "data": datetime.now().isoformat(timespec="seconds"),
                    # koszt zapisuję razem ze zgłoszeniem, przyda się w historii zgłoszeń
                    "koszt_usd": round(koszt_zgloszenia_usd(), 5),
                },
            )
        ],
    )

    return point_id


# Wyszukiwanie jak list_notes_from_db w "Audio Notatkach" (moduł 7, wersja v6).
# Szukam tylko w historii, bo tylko tam jest przyczyna i to, co wtedy pomogło -
# świeże zgłoszenia z aplikacji jeszcze tego nie mają.
def find_similar_failures(query, limit=5, do_zgloszenia=True):
    qdrant_client = get_qdrant_client()
    awarie = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        query=get_embedding(text=query, etap="podobne awarie", do_zgloszenia=do_zgloszenia),
        query_filter=Filter(
            must=[FieldCondition(key="zrodlo", match=MatchValue(value="historia"))]
        ),
        limit=limit,
    ).points

    result = []
    for awaria in awarie:
        result.append({
            **awaria.payload,
            "score": awaria.score,
        })

    return result


# Lista zgłoszeń bez wyszukiwania - scroll jak w list_notes_from_db z "Audio Notatek"
# (moduł 7, wersja v6). Tu biorę tylko zgłoszenia z aplikacji i wszystkie naraz,
# a nie 10, bo filtruję je dopiero w tabeli.
def list_tickets_from_db():
    qdrant_client = get_qdrant_client()
    zgloszenia = qdrant_client.scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="zrodlo", match=MatchValue(value="aplikacja"))]
        ),
        limit=1000,
        with_vectors=False,
    )[0]

    result = []
    for zgloszenie in zgloszenia:
        result.append({
            "nr": zgloszenie.id,
            **zgloszenie.payload,
        })

    return result


# Jedna awaria z historii w ramce - używam tego w obu zakładkach
def show_failures(awarie):
    if not awarie:
        st.info("Nie znalazłem podobnych awarii w historii.")

    for awaria in awarie:
        with st.container(border=True):
            st.markdown(f'**{awaria["urzadzenie"]}** · {awaria["system"]} · {awaria["lokalizacja"]}')
            st.markdown(f':violet[podobieństwo: {awaria["score"]:.2f}]')
            st.markdown(f'**Objaw:** {awaria["objaw"]}')
            if awaria["kod_alarmu"]:
                st.markdown(f'**Kod alarmu:** {awaria["kod_alarmu"]}')
            st.markdown(f'**Przyczyna:** {awaria["przyczyna"]}')
            st.markdown(f'**Co pomogło:** {awaria["dzialanie_naprawcze"]}')
            st.caption(
                f'Przestój: {awaria["czas_przestoju_h"]:g} h · priorytet: {awaria["priorytet"]} · '
                f'data: {awaria["data"][:10]}'
            )


#
# MAIN
#
st.set_page_config(page_title="Głosowe zgłaszanie awarii", layout="centered")

st.title("Głosowe zgłaszanie awarii")

# Ochrona klucza OpenAI - przepisane z "Audio Notatek" (moduł 7, wersja v3).
# Jest klucz w .env -> biorę go. Nie ma -> użytkownik wpisuje własny,
# a ja trzymam go tylko w session_state (znika po zamknięciu karty).
# Komunikat o złym kluczu wziąłem ze swojego zadania domowego z modułu 9 (półmaraton)
if st.session_state.get("bad_api_key"):
    st.error("Nieprawidłowy klucz API OpenAI. Sprawdź go i wpisz ponownie.")

if not st.session_state.get("openai_api_key"):
    # Klucza z .env nie biorę drugi raz, jeśli OpenAI już go odrzucił
    if "OPENAI_API_KEY" in env and not st.session_state.get("bad_api_key"):
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password", key="pole_klucza")
        if st.session_state["openai_api_key"]:
            # nowy klucz wpisany - chowam komunikat o złym kluczu
            st.session_state["bad_api_key"] = False
            st.rerun()

# Bez klucza nie ma co pokazywać reszty aplikacji
if not st.session_state.get("openai_api_key"):
    st.stop()

# Klucz sprawdzam raz, od razu po wpisaniu, a nie dopiero przy pierwszym zapytaniu jak w module 9.
# Tu zapytań do OpenAI jest pięć rodzajów, a instructor opakowuje błąd złego klucza w swój
# wyjątek, więc łapanie AuthenticationError przy każdym wywołaniu by nie wystarczyło.
# models.list() nic nie kosztuje, a przy złym kluczu rzuca AuthenticationError.
if not st.session_state.get("klucz_sprawdzony"):
    try:
        get_openai_client().models.list()
        st.session_state["klucz_sprawdzony"] = True
    except AuthenticationError:
        st.session_state["openai_api_key"] = None
        st.session_state["bad_api_key"] = True
        # czyszczę pole, żeby zły klucz w nim nie został
        st.session_state["pole_klucza"] = ""
        st.rerun()

# Streamlit przy każdym kliknięciu uruchamia skrypt od nowa,
# więc nagranie i transkrypcję trzymam w session_state (moduł 6, lekcja o stanie)
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None

if "audio_nazwa" not in st.session_state:
    st.session_state["audio_nazwa"] = None

if "audio_bytes_md5" not in st.session_state:
    st.session_state["audio_bytes_md5"] = None

if "transkrypcja" not in st.session_state:
    st.session_state["transkrypcja"] = ""

# zgloszenie = propozycja modelu, zgloszenie_zatwierdzone = to, co zatwierdził człowiek
if "zgloszenie" not in st.session_state:
    st.session_state["zgloszenie"] = None

if "zgloszenie_zatwierdzone" not in st.session_state:
    st.session_state["zgloszenie_zatwierdzone"] = None

# numer zgłoszenia w bazie - pamiętam go, żeby poprawka nadpisała ten sam rekord
if "zgloszenie_id" not in st.session_state:
    st.session_state["zgloszenie_id"] = None

# podobne awarie do zatwierdzonego zgłoszenia - trzymam wynik, żeby nie liczyć
# embeddingu od nowa przy każdym odświeżeniu strony
if "podobne_awarie" not in st.session_state:
    st.session_state["podobne_awarie"] = None

# zdjęcie tabliczki / HMI jest opcjonalne; md5 tak samo jak przy nagraniu
if "zdjecie_md5" not in st.session_state:
    st.session_state["zdjecie_md5"] = None

if "dane_ze_zdjecia" not in st.session_state:
    st.session_state["dane_ze_zdjecia"] = None

# koszty liczę jak w module 6: osobno bieżące zgłoszenie, osobno cała sesja
if "koszty_zgloszenia" not in st.session_state:
    st.session_state["koszty_zgloszenia"] = []

if "koszt_sesji_usd" not in st.session_state:
    st.session_state["koszt_sesji_usd"] = 0.0

with st.sidebar:
    koszty_placeholder = st.empty()
pokaz_koszty()

# Uśpiony klaster w chmurze albo zły adres dawał mi długi, nieczytelny błąd,
# więc pokazuję zwykły komunikat i zatrzymuję aplikację.
try:
    assure_db_collection_exists()
except Exception as e:
    st.error(
        "Nie mogę połączyć się z bazą Qdrant. Sprawdź, czy klaster w Qdrant Cloud działa "
        "oraz czy QDRANT_URL i QDRANT_API_KEY są poprawne."
    )
    st.caption(f"Szczegóły: {type(e).__name__}: {e}")
    st.stop()

# Zakładki jak w "Audio Notatkach" (dodawanie i wyszukiwanie), u mnie jeszcze historia zgłoszeń
add_tab, search_tab, history_tab = st.tabs(["Zgłoś awarię", "Szukaj w historii", "Historia zgłoszeń"])

# Wyszukiwarkę i historię rysuję najpierw. Zakładka zgłoszenia zatrzymuje skrypt (st.stop),
# gdy brakuje kolejnego kroku, i wszystko, co byłoby niżej, już by się nie pokazało.
with search_tab:
    query = st.text_input("Opisz awarię własnymi słowami", placeholder="np. pompa obiegowa hałasuje")
    if st.button("Szukaj") and query:
        with st.spinner("Szukam podobnych awarii..."):
            show_failures(find_similar_failures(query, do_zgloszenia=False))

with history_tab:
    zgloszenia_df = pd.DataFrame(list_tickets_from_db())

    if zgloszenia_df.empty:
        st.info("Nie ma jeszcze zatwierdzonych zgłoszeń.")
    else:
        # Starsze zgłoszenia (sprzed v8) nie mają kosztu - pandas wstawi tam puste pole
        if "koszt_usd" not in zgloszenia_df.columns:
            zgloszenia_df["koszt_usd"] = None
        zgloszenia_df["data"] = pd.to_datetime(zgloszenia_df["data"])
        zgloszenia_df = zgloszenia_df.sort_values("data", ascending=False)

        # Filtry - puste pole wyboru znaczy "wszystkie"
        c0, c1, c2 = st.columns(3)
        with c0:
            filtr_system = st.multiselect("System", list(get_args(SYSTEMY)), placeholder="wszystkie")
        with c1:
            filtr_priorytet = st.multiselect("Priorytet", list(get_args(PRIORYTETY)), placeholder="wszystkie")
        with c2:
            filtr_gmp = st.multiselect("Wpływ na GMP", list(get_args(WPLYWY)), placeholder="wszystkie")

        c0, c1 = st.columns(2)
        with c0:
            filtr_tekst = st.text_input("Szukaj w urządzeniu, lokalizacji i objawie", placeholder="np. AHU-03")
        with c1:
            filtr_daty = st.date_input(
                "Data zgłoszenia (od - do)",
                value=(zgloszenia_df["data"].min().date(), zgloszenia_df["data"].max().date()),
            )

        wynik = zgloszenia_df
        if filtr_system:
            wynik = wynik[wynik["system"].isin(filtr_system)]
        if filtr_priorytet:
            wynik = wynik[wynik["priorytet"].isin(filtr_priorytet)]
        if filtr_gmp:
            wynik = wynik[wynik["wplyw_na_gmp"].isin(filtr_gmp)]
        if filtr_tekst:
            tekst = wynik["urzadzenie"] + " " + wynik["lokalizacja"] + " " + wynik["objaw"]
            wynik = wynik[tekst.str.contains(filtr_tekst, case=False, regex=False)]
        # Przy wybieraniu zakresu date_input przez chwilę ma tylko datę początkową
        if len(filtr_daty) == 2:
            od, do = filtr_daty
            wynik = wynik[(wynik["data"].dt.date >= od) & (wynik["data"].dt.date <= do)]

        # Lista braków do tabeli i CSV jako jeden tekst, tak samo jak przy pobieraniu jednego zgłoszenia
        wynik = wynik.assign(braki_informacji=wynik["braki_informacji"].apply("; ".join))
        kolumny = [
            "nr", "data", "system", "urzadzenie", "lokalizacja", "kod_alarmu", "priorytet",
            "wplyw_na_bezpieczenstwo", "wplyw_na_gmp", "objaw", "dzialanie_natychmiastowe",
            "braki_informacji", "koszt_usd",
        ]
        wynik = wynik[kolumny]

        c0, c1 = st.columns(2)
        with c0:
            st.metric("Zgłoszeń", f"{len(wynik)} z {len(zgloszenia_df)}")
        with c1:
            st.metric("Koszt tych zgłoszeń", f'${wynik["koszt_usd"].sum():.4f}')

        st.dataframe(
            wynik,
            hide_index=True,
            column_config={
                "data": st.column_config.DatetimeColumn("data", format="YYYY-MM-DD HH:mm"),
                "koszt_usd": st.column_config.NumberColumn("koszt USD", format="%.4f"),
            },
        )

        st.download_button(
            "Pobierz CSV",
            data=wynik.to_csv(index=False).encode("utf-8-sig"),
            file_name="historia_zgloszen.csv",
            mime="text/csv",
            key="pobierz_historie_csv",
        )

with add_tab:
    st.header("1. Opis awarii")

    # W kursie było samo nagrywanie. Dodałem wgrywanie pliku, bo w hałasie
    # maszynowni wygodniej nagrać opis telefonem i wrzucić go później.
    zrodlo = st.radio(
        "Skąd biorę opis awarii?",
        ["Nagrywam teraz", "Wgrywam plik audio"],
        horizontal=True,
    )

    if zrodlo == "Nagrywam teraz":
        # Nagrywarka z "Audio Notatek" (streamlit-audiorecorder)
        nagranie = audiorecorder(
            start_prompt="Nagraj opis awarii",
            stop_prompt="Zatrzymaj nagrywanie",
        )
        if nagranie:
            # Nagrywarka oddaje AudioSegment z pydub, a ja potrzebuję bajtów mp3
            audio = BytesIO()
            nagranie.export(audio, format="mp3")
            st.session_state["audio_bytes"] = audio.getvalue()
            st.session_state["audio_nazwa"] = "nagranie.mp3"
        else:
            st.session_state["audio_bytes"] = None
            st.session_state["audio_nazwa"] = None

    else:
        plik = st.file_uploader("Plik z opisem awarii", type=["mp3", "wav", "m4a", "ogg"])
        if plik:
            st.session_state["audio_bytes"] = plik.getvalue()
            st.session_state["audio_nazwa"] = plik.name
        else:
            st.session_state["audio_bytes"] = None
            st.session_state["audio_nazwa"] = None

    # Sztuczka z md5 z "Audio Notatek": po skrócie poznaję, że nagranie się zmieniło.
    # Stara transkrypcja już do niego nie pasuje, więc ją czyszczę.
    if st.session_state["audio_bytes"]:
        current_md5 = md5(st.session_state["audio_bytes"]).hexdigest()
    else:
        current_md5 = None

    if st.session_state["audio_bytes_md5"] != current_md5:
        st.session_state["audio_bytes_md5"] = current_md5
        st.session_state["transkrypcja"] = ""
        st.session_state["zgloszenie"] = None
        st.session_state["zgloszenie_zatwierdzone"] = None
        st.session_state["zgloszenie_id"] = None
        st.session_state["podobne_awarie"] = None
        # nowe nagranie = nowe zgłoszenie, więc licznik kosztu zgłoszenia od zera
        st.session_state["koszty_zgloszenia"] = []
        pokaz_koszty()

    # Bez nagrania dalsze kroki nie mają sensu
    if not st.session_state["audio_bytes"]:
        st.info("Nagraj opis awarii albo wgraj plik audio.")
        st.stop()

    st.audio(st.session_state["audio_bytes"])
    st.caption(f"Plik: {st.session_state['audio_nazwa']}")

    # Zdjęcie jest dodatkiem: kod alarmu z ekranu albo typ z tabliczki łatwiej sfotografować niż podyktować
    zdjecie = st.file_uploader(
        "Zdjęcie tabliczki znamionowej lub ekranu HMI (opcjonalnie)",
        type=["png", "jpg", "jpeg"],
    )
    if zdjecie:
        zdjecie_md5 = md5(zdjecie.getvalue()).hexdigest()
    else:
        zdjecie_md5 = None

    # Inne zdjęcie (albo jego brak) = stare odczyty już nie pasują
    if st.session_state["zdjecie_md5"] != zdjecie_md5:
        st.session_state["zdjecie_md5"] = zdjecie_md5
        st.session_state["dane_ze_zdjecia"] = None

    if zdjecie:
        st.image(zdjecie, width=300)
        # Odczyt dopiero po kliknięciu, bo analiza zdjęcia kosztuje
        if st.button("Odczytaj dane ze zdjęcia"):
            with st.spinner("Czytam zdjęcie..."):
                st.session_state["dane_ze_zdjecia"] = read_image_data(zdjecie.getvalue(), zdjecie.type)

    if st.session_state["dane_ze_zdjecia"]:
        st.caption("Odczytane ze zdjęcia - trafi do zgłoszenia razem z transkrypcją:")
        st.json(st.session_state["dane_ze_zdjecia"])

    st.header("2. Transkrypcja")

    # Transkrypcję robię dopiero po kliknięciu, żeby nie płacić za każde odświeżenie strony
    if st.button("Transkrybuj nagranie"):
        with st.spinner("Zamieniam nagranie na tekst..."):
            st.session_state["transkrypcja"] = transcribe_audio(
                st.session_state["audio_bytes"],
                st.session_state["audio_nazwa"],
            )

    # W hałasie whisper potrafi przekręcić nazwę urządzenia albo kod alarmu,
    # dlatego tekst zawsze można poprawić przed kolejnym krokiem.
    # Pole jest spięte z session_state przez key, więc poprawki nie znikają.
    if st.session_state["transkrypcja"]:
        st.text_area(
            "Transkrypcja (popraw, jeśli model coś przekręcił)",
            key="transkrypcja",
            height=200,
        )

    # Bez transkrypcji nie mam z czego zrobić zgłoszenia
    if not st.session_state["transkrypcja"]:
        st.stop()

    st.header("3. Zgłoszenie")

    if st.button("Utwórz zgłoszenie z transkrypcji"):
        with st.spinner("Układam zgłoszenie..."):
            st.session_state["zgloszenie"] = create_ticket(
                st.session_state["transkrypcja"],
                st.session_state["dane_ze_zdjecia"],
            )
            # nowa propozycja modelu unieważnia wcześniejsze zatwierdzenie
            st.session_state["zgloszenie_zatwierdzone"] = None
            st.session_state["zgloszenie_id"] = None
            st.session_state["podobne_awarie"] = None

    if not st.session_state["zgloszenie"]:
        st.stop()

    zgloszenie = st.session_state["zgloszenie"]

    st.caption("Priorytet i wpływ na GMP to propozycja modelu. Sprawdź je przed zatwierdzeniem.")

    # Formularz z st.form, żeby strona nie odświeżała się po każdej zmianie pola,
    # tylko raz - po kliknięciu "Zatwierdź"
    with st.form("formularz_zgloszenia"):
        systemy = list(get_args(SYSTEMY))
        priorytety = list(get_args(PRIORYTETY))
        wplywy = list(get_args(WPLYWY))

        system = st.selectbox("System", systemy, index=systemy.index(zgloszenie["system"]))
        urzadzenie = st.text_input("Urządzenie", value=zgloszenie["urzadzenie"])
        lokalizacja = st.text_input("Lokalizacja", value=zgloszenie["lokalizacja"])
        objaw = st.text_area("Objaw", value=zgloszenie["objaw"])
        kod_alarmu = st.text_input("Kod alarmu", value=zgloszenie["kod_alarmu"])

        c0, c1, c2 = st.columns(3)
        with c0:
            priorytet = st.selectbox(
                "Priorytet", priorytety, index=priorytety.index(zgloszenie["priorytet"])
            )
        with c1:
            wplyw_na_bezpieczenstwo = st.selectbox(
                "Wpływ na bezpieczeństwo",
                wplywy,
                index=wplywy.index(zgloszenie["wplyw_na_bezpieczenstwo"]),
            )
        with c2:
            wplyw_na_gmp = st.selectbox(
                "Wpływ na GMP", wplywy, index=wplywy.index(zgloszenie["wplyw_na_gmp"])
            )

        dzialanie_natychmiastowe = st.text_area(
            "Działanie natychmiastowe", value=zgloszenie["dzialanie_natychmiastowe"]
        )
        # Listę braków pokazuję jako tekst, jeden brak w linii - łatwo coś skreślić albo dopisać
        braki_informacji = st.text_area(
            "Braki informacji (jeden w linii)",
            value="\n".join(zgloszenie["braki_informacji"]),
        )

        zatwierdzone = st.form_submit_button("Zatwierdź zgłoszenie")

    if zatwierdzone:
        st.session_state["zgloszenie_zatwierdzone"] = {
            "system": system,
            "urzadzenie": urzadzenie,
            "lokalizacja": lokalizacja,
            "objaw": objaw,
            "kod_alarmu": kod_alarmu,
            "priorytet": priorytet,
            "wplyw_na_bezpieczenstwo": wplyw_na_bezpieczenstwo,
            "wplyw_na_gmp": wplyw_na_gmp,
            "dzialanie_natychmiastowe": dzialanie_natychmiastowe,
            "braki_informacji": [b.strip() for b in braki_informacji.splitlines() if b.strip()],
        }
        # Do bazy trafia dopiero to, co zatwierdził człowiek - nigdy sama propozycja modelu
        with st.spinner("Zapisuję zgłoszenie w bazie..."):
            st.session_state["zgloszenie_id"] = add_ticket_to_db(
                st.session_state["zgloszenie_zatwierdzone"],
                point_id=st.session_state["zgloszenie_id"],
            )
        st.toast("Zgłoszenie zapisane w bazie", icon="🎉")
        # Od razu szukam podobnych awarii - po tym samym tekście, z którego liczę embedding zgłoszenia
        with st.spinner("Szukam podobnych awarii..."):
            st.session_state["podobne_awarie"] = find_similar_failures(
                ticket_to_text(st.session_state["zgloszenie_zatwierdzone"])
            )
        # Zakładka z historią zgłoszeń jest rysowana wyżej, jeszcze przed zapisem.
        # Odświeżam stronę, żeby nowe zgłoszenie od razu było też na liście.
        st.rerun()

    if st.session_state["zgloszenie_zatwierdzone"]:
        zgloszenie_zatwierdzone = st.session_state["zgloszenie_zatwierdzone"]
        st.success(
            f'Zgłoszenie nr {st.session_state["zgloszenie_id"]} zatwierdzone i zapisane w bazie '
            f"(w bazie jest teraz zgłoszeń: {count_tickets_in_db()})"
        )
        st.json(zgloszenie_zatwierdzone)

        # W CSV lista braków musi być jednym polem, więc sklejam ją średnikami.
        # utf-8-sig, żeby Excel dobrze pokazał polskie znaki.
        wiersz = dict(zgloszenie_zatwierdzone)
        wiersz["braki_informacji"] = "; ".join(wiersz["braki_informacji"])

        c0, c1 = st.columns(2)
        with c0:
            st.download_button(
                "Pobierz JSON",
                data=json.dumps(zgloszenie_zatwierdzone, ensure_ascii=False, indent=2),
                file_name="zgloszenie.json",
                mime="application/json",
            )
        with c1:
            st.download_button(
                "Pobierz CSV",
                data=pd.DataFrame([wiersz]).to_csv(index=False).encode("utf-8-sig"),
                file_name="zgloszenie.csv",
                mime="text/csv",
            )

        st.header("4. Podobne awarie z historii")
        st.caption("Dane w historii są syntetyczne - to pokaz, jak działa wyszukiwanie.")
        show_failures(st.session_state["podobne_awarie"] or [])

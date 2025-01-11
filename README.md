# Projekt końcowy na przedmiot Zastosowanie Uczenia Maszynowego (ZUM) 


### 🔍 **Opis**  
Projekt wykorzystuje dane z zestawu "CNN/Daily Mail" w celu trenowania modelu T5 do generowania streszczeń artykułów. Obejmuje on przygotowania danych, czyszczenie, tokenizację, oraz trening modelu i generowanie prognoz. Celem jest stworzenie modelu do automatycznego streszczania tekstów.
***
### 👨‍💻 **Zespół**
- Maria Fuk s20436
- Jakub Augustyniak s20131
- Piotr Kaczmarek s30216

---
### 📚 **Wykorzystane narzędzia**:
- **Hugging Face Transformers**: Ładowanie i trening modelu T5.
- **PyTorch**: Framework do uczenia maszynowego i głębokiego uczenia, wykorzystany do trenowania modelu.
- **Datasets**: Załadowanie i przetwarzanie datasetu.
- **Matplotlib**: Wizualizacja wyników.
---
### Struktura projektu

1. **Przygotowanie danych**:
   - Wczytanie zestawu danych `cnn_dailymail` w wersji `3.0.0`.
   - Konwersja danych do formatu Pandas DataFrame.
   - Analiza długości artykułów oraz ich streszczeń.

2. **Wstępna analiza danych**:
   - Średnia długość artykułów i streszczeń.
   - Tworzenie histogramów pokazujących rozkład długości tekstów i streszczeń.

3. **Czyszczenie danych**:
   - Usuwanie zbędnych linków URL z artykułów i streszczeń.
   - Normalizacja tekstu (usuwanie nadmiarowych białych znaków).

4. **Tokenizacja i przygotowanie danych**:
   - Tokenizacja artykułów oraz streszczeń za pomocą tokenizera modelu T5.
   - Przygotowanie danych wejściowych i etykiet do treningu.

5. **Trening modelu**:
   - Użycie modelu T5 (wersja `t5-small`).
   - Ustawienie parametrów treningu.
   - Trening modelu na danych treningowych i walidacyjnych.

6. **Ocena modelu**:
   - Ocena wyników na zbiorach walidacyjnym i testowym.
   - Generowanie prognoz na podstawie testowych danych.

7. **Zapis modelu**:
   - Zapisanie wytrenowanego modelu oraz tokenizera na dysku.

## Jak uruchomić projekt

1. **Instalacja wymaganych bibliotek**:
   Zainstaluj wszystkie wymagane biblioteki używając pip:

   ```bash
   pip install transformers torch datasets matplotlib accelerate

2. **Uruchomienie skryptu**:
Skrypt możesz uruchomić w dowolnym środowisku Python, takim jak:
- Jupyter Notebook
- Google Colab
- Lokalna instalacja Python z użyciem IDE (np. PyCharm, VSCode)
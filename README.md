
# LEGO Dataanalyseprosjekt

Denne mappen inneholder prosjektleveransen for emnet **Statistisk læring og Data Science** (ISTx1003) ved NTNU, høsten 2024. Prosjektet fokuserer på anvendelse av regresjon, klassifikasjon og klyngeanalyse for å analysere LEGO-datasett.

## Innholdsfortegnelse
- [Prosjektoversikt](#prosjektoversikt)
- [Datasettbeskrivelse](#datasettbeskrivelse)
- [Mappestruktur](#mappestruktur)
- [Installasjon og oppsett](#installasjon-og-oppsett)
- [Bruksanvisning](#bruksanvisning)
- [Oppgavebeskrivelse](#oppgavebeskrivelse)
- [Referanser](#referanser)

---

## Prosjektoversikt
Prosjektet består av tre hoveddeler:
1. **Regresjonsanalyse** (50%): Bruk av multippel lineær regresjon for å utforske forholdet mellom egenskaper ved LEGO-sett (f.eks. antall brikker) og pris.
2. **Klassifikasjon** (30%): Implementasjon av klassifikasjonsmetoder på datasettet.
3. **Klyngeanalyse** (20%): Bruk av klyngeanalyse for å identifisere mønstre i datasettet.

### Mål
Formålet med prosjektet er å bruke statistiske og maskinlæringsbaserte metoder for å trekke innsikter fra datasettet og løse problemstillinger basert på definert pipeline og metodologi.

---

## Datasettbeskrivelse
Datasettet er hentet fra artikkelen *Building a Multiple Linear Regression Model With LEGO Brick Data* av Peterson og Ziegler (2021). Det består av 1304 observasjoner samlet fra 1. januar 2018 til 11. september 2020, og inkluderer følgende variabler:
- **Set Name**: Navn på LEGO-settet
- **Theme**: Tema settet tilhører
- **Pieces**: Antall brikker i settet
- **Price**: Pris i dollar
- **Pages**: Antall sider i byggeinstruksjonen
- **Unique Pieces**: Antall unike brikker

Datapreprosessering, inkludert opprensking av data og omkoding, er utført i `main.py` og i oppgave-notatbøkene.

---

## Mappestruktur
Følgende filer og mapper er inkludert i prosjektet:

```
/lego-data-analysis
│
├── oppg1.ipynb           # Notatbok for Oppgave 1: Multippel lineær regresjon
├── oppg2.ipynb           # Notatbok for Oppgave 2: Klassifikasjon
├── oppg3.ipynb           # Notatbok for Oppgave 3: Klyngeanalyse
├── main.py               # Python-script for opprensking og visualisering av data
├── lego.population.csv   # LEGO-datasettet
└── README.md             # Denne filen
```

---

## Installasjon og oppsett
1. Klon eller last ned dette prosjektet fra GitHub.
2. Installer nødvendige biblioteker med følgende kommando:
   ```bash
   pip install -r requirements.txt
   ```
   **Krav**:
   - Python 3.8 eller nyere
   - Nødvendige pakker: `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`, `scipy`

3. Last opp datasettet `lego.population.csv` til riktig katalog.

---

## Bruksanvisning
### Kjøring av skript:
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Åpne og kjør de respektive `.ipynb`-filene for oppgavene.
3. For å kjøre dataopprensking og visualisering separat, bruk:
   ```bash
   python main.py
   ```

---

## Oppgavebeskrivelse
### Oppgave 1: Multippel lineær regresjon
- Definer en problemstilling, for eksempel:
  - "Er LEGO for gutter dyrere enn LEGO for jenter?"
- Analyser datasettet ved hjelp av multippel lineær regresjon.
- Resultatene leveres som en rapport (maks 5 sider).

### Oppgave 2: Klassifikasjon
- Implementer klassifikasjonsmodeller for datasettet.
- Svarene skal legges inn i en forhåndsdefinert mal.

### Oppgave 3: Klyngeanalyse
- Bruk klyngeanalyse for å identifisere mønstre i LEGO-temaene.
- Resultatene skal dokumenteres i samme mal som oppgave 2.

---

## Referanser
- Peterson, A. D., & Ziegler, L. (2021). *Building a Multiple Linear Regression Model With LEGO Brick Data*. Journal of Statistics and Data Science Education, 29(3), 297–303.
- NTNU ISTx1003 Prosjektbeskrivelse (2024).

---

Denne README gir en oversikt over prosjektet og hvordan filene skal brukes. For spørsmål, ta kontakt via kursplattformen eller veiledere.

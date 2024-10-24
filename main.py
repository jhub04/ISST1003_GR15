import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("lego.population.csv", sep = ",", encoding = "latin1")

# fjerner forklaringsvariabler vi ikke trenger
df2 = df[['Set_Name', 'Theme', 'Pieces', 'Price', 'Pages',  'Unique_Pieces']]

# fjerner observasjoner med manglende datapunkter
df2 = df2.dropna()

# gjør themes om til string og fjern alle tegn vi ikke vil ha med
df2['Theme'] = df2['Theme'].astype(str)
df2['Theme'] = df2['Theme'].str.replace(r'[^a-zA-Z0-9\s-]', '', regex = True)

# fjerner dollartegn og trademark-tegn fra datasettet
df2['Price'] = df2['Price'].str.replace('\$', '', regex = True)

# og gjør så prisen om til float
df2['Price'] = df2['Price'].astype(float)

# det er dataset dere skal bruke!
print(df2)

print(df2.mean(numeric_only = True))
# vil beregne gjennomsnittet for alle kolonnene i DataFrame df2 som inneholder numeriske verdier. Parametret numeric_only=True sørger for at funksjonen ignorerer ikke-numeriske kolonner (som strenger eller kategoriske variabler).

print(df2['Theme'].value_counts())
#Finner ut hvor mange tilfeller det er av hver type theme som er representert i datasettet

# Lager et histogram for kolonnen 'Price' fra df2
plt.hist(df2['Price'], bins=20, color='skyblue', edgecolor='black')  # Lager et histogram med 20 bins, søylefarge lyseblå og kantfarge svart

# Setter etikett for x-aksen
plt.xlabel('Pris i dollar [$]')  # X-aksen beskriver prisen i dollar

# Setter etikett for y-aksen
plt.ylabel('')  # Y-aksen er tom (ingen beskrivelse)

# Justerer forholdet mellom x- og y-aksen til 1:1
plt.gca().set_aspect(1)  # Setter forholdet mellom aksene til 1:1

# Viser histogrammet
plt.show()  # Viser plottet på skjermen

# Lager et spredningsdiagram (scatter plot) mellom antall brikker og pris
plt.scatter(df2['Pieces'], df2['Price'])  # Lager et scatter plot hvor x-aksen representerer 'Pieces' (antall brikker) og y-aksen representerer 'Price' (pris)

# Setter etikett for x-aksen
plt.xlabel('Antall brikker')  # X-aksen viser antall brikker

# Setter etikett for y-aksen
plt.ylabel('Pris i dollar [$]')  # Y-aksen viser prisen i dollar

# Justerer forholdet mellom x- og y-aksen til 5:1
plt.gca().set_aspect(5)  # Setter forholdet mellom aksene slik at y-aksen er 5 ganger kortere enn x-aksen

# Viser plottet
plt.show()  # Viser spredningsdiagrammet på skjermen

# hva er det dyreste settet i datasettet mon tro?
print(df2.loc[df2['Price'].idxmax()])
# og hvilket har flest brikker?
print(df2.loc[df2['Pieces'].idxmax()])

# hvilke tema har de billigste settene?
df2.groupby('Theme')['Price'].mean().sort_values(ascending=True)[:3]

# hvilke tema har flest brikker?
df2.groupby('Theme')['Pieces'].mean().sort_values(ascending=False)[:3]

sns.pairplot(df2, vars = ['Price', 'Pieces', 'Pages', 'Unique_Pieces'],
             hue = 'Theme',
             diag_kind = 'kde',
             plot_kws = dict(alpha = 0.4))
plt.show()

"""
# Lager et parvis plott (pairplot) for de angitte variablene i df2
sns.pairplot(df2, 
             vars=['Price', 'Pieces', 'Pages', 'Unique_Pieces'],  # Variablene som skal plottes parvis mot hverandre
             hue='Theme',  # Farger punktene i plottet basert på verdiene i kolonnen 'Theme'
             diag_kind='kde',  # Bruker en kjernetetthetsestimering (kde) for diagonale plot (univariate fordeling)
             plot_kws=dict(alpha=0.4))  # Setter alfa-verdien til 0.4 for å gjøre punktene mer gjennomsiktige i scatter plottet

# Viser plottet
plt.show()  # Viser plottet på skjermen

"""

# Enkel lineær regresjon
formel = 'Price ~ Pieces'  # Spesifiserer formelen for lineær regresjon, der 'Price' er avhengig variabel og 'Pieces' er uavhengig variabel

# Lager en OLS (ordinary least squares) lineær regresjonsmodell basert på formelen
modell = smf.ols(formel, data=df2)  # OLS-modellen bygges med formelen 'Price ~ Pieces' og dataen fra DataFrame 'df2'

# Tilpasser modellen til dataene
resultat = modell.fit()  # Tilpasser (fit) modellen til dataene for å finne de beste parameterne

# Viser en detaljert oppsummering av regresjonsresultatene
resultat.summary()  # Skriver ut en statistisk oppsummering av regresjonsmodellen, inkludert koeffisienter, p-verdier, R-kvadrat, og andre statistiske mål


# Henter regresjonskoeffisienten (stigningstall/slope) og interceptet fra resultatet av den lineære regresjonen
slope = resultat.params['Pieces']  # Stigningstallet for antall brikker (Pieces) fra regresjonsmodellen
intercept = resultat.params['Intercept']  # Intercept (skjæringspunktet med y-aksen) fra regresjonsmodellen

# Lager en numpy-array fra kolonnen 'Pieces' for regresjonsprediksjoner
regression_x = np.array(df2['Pieces'])  # Henter verdiene for antall brikker fra df2 og lagrer dem som en numpy-array

# Beregner de estimerte verdiene for pris basert på regresjonsmodellen
regression_y = slope * regression_x + intercept  # Bruker ligningen for en rett linje (y = mx + b) for å beregne pris

# Lager et scatter plot som viser de faktiske datapunktene
plt.scatter(df2['Pieces'], df2['Price'], label='Data Points')  # Plotter de faktiske dataene for antall brikker og pris

# Tegner regresjonslinjen over scatter plottet
plt.plot(regression_x, regression_y, color='red', label='Regression Line')  # Tegner regresjonslinjen i rødt, som viser den estimerte lineære sammenhengen

# Setter etiketten for x-aksen
plt.xlabel('Antall brikker')  # X-aksen representerer antall brikker (Pieces)

# Setter etiketten for y-aksen
plt.ylabel('Pris [$]')  # Y-aksen representerer pris i dollar (Price)

# Setter tittelen på plottet
plt.title('Kryssplott med regresjonslinje (enkel LR)')  # Tittelen beskriver plottet som et kryssplott med en enkel lineær regresjonslinje

# Viser legenden som forklarer de to plottelementene (datapunkter og regresjonslinje)
plt.legend()  # Viser en forklaring for datapunktene og regresjonslinjen i plottet

# Slår på grid (rutenett) i plottet for bedre lesbarhet
plt.grid()  # Legger til et rutenett for å gjøre plottet lettere å lese

# Viser plottet på skjermen
plt.show()  # Viser plottet med datapunktene og regresjonslinjen


#lager en subplot med 2 grafer, 1 rad 2 kolonner og sjekker om variansen i residualene er uavhengig for kovariatet, svaret er nei fordi det er ikke spredt gjevnt i den første grafen, og den holder seg ikke nærme den røde linja i den andre grafen. 

figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = resultat.fittedvalues, y = resultat.resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

sm.qqplot(resultat.resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.show()

# Velger ut et subset av dataene basert på spesifikke temaer
mythemes = ['Star Wars', 'NINJAGO', 'Harry Potter']  # Lager en liste over de temaene vi er interessert i

# Lager et subset av df2 der 'Theme'-kolonnen inneholder kun de valgte temaene
subset_df = df2[df2['Theme'].isin(mythemes)]  # Finner radene i df2 der 'Theme' er enten 'Star Wars', 'NINJAGO', eller 'Harry Potter'

# Lager et parvis plott (pairplot) for de valgte variablene i subset_df
sns.pairplot(subset_df, 
             vars=['Price', 'Pieces', 'Pages', 'Unique_Pieces'],  # Spesifiserer variablene som skal plottes parvis mot hverandre
             hue='Theme',  # Farger punktene basert på verdiene i kolonnen 'Theme'
             diag_kind='kde',  # Bruker en kjernetetthetsestimering (kde) for de diagonale plottene (univariate fordeling)
             plot_kws=dict(alpha=0.4))  # Setter alfa-verdien til 0.4 for å gjøre punktene mer gjennomsiktige i scatter plottet

# Viser plottet
plt.show()  # Viser parvise plottet på skjermen

# enkel lineær regresjon, tar ikke hensyn til tema
res_sub = smf.ols('Price ~ Pieces' , data = subset_df).fit()

# enkel LR for hvert tema hver for seg
resultater = []
for i, theme in enumerate(mythemes):
    modell3 = smf.ols('Price ~ Pieces' , data = subset_df[subset_df['Theme'].isin([theme])])
    resultater.append(modell3.fit())

# plott av dataene og regresjonslinjene
for i, theme in enumerate(mythemes):
    slope = resultater[i].params['Pieces']
    intercept = resultater[i].params['Intercept']

    regression_x = np.array(subset_df[subset_df['Theme'].isin([theme])]['Pieces'])
    regression_y = slope * regression_x + intercept

    # Plot scatter plot and regression line
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin([theme])]['Price'], color=plt.cm.tab10(i))
    plt.plot(regression_x, regression_y, color=plt.cm.tab10(i), label=theme)
    
plt.xlabel('Antall brikker')
plt.ylabel('Pris')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()

# multippel lineær regresjon
modell3_mlr = smf.ols('Price ~ Pieces + Theme' , data = subset_df)
modell3_mlr.fit().summary()

"""
Basert på resultatene i bildet, ser vi koeffisienter for Theme[T.NINJAGO] og Theme[T.Star Wars],
men ikke for Theme[Harry Potter]. Dette betyr at "Harry Potter" fungerer som referansekategorien i denne modellen.
Koeffisientene for T.NINJAGO og T.Star Wars viser hvor mye gjennomsnittsprisen for disse temaene avviker fra gjennomsnittsprisen for "Harry Potter"-settene, gitt at antall brikker holdes konstant.

"""

# multippel lineær regresjon med en annen referansekategori - fordi harry potter var det før
modell3_mlr_alt = smf.ols('Price ~ Pieces + C(Theme, Treatment("Star Wars"))' , data = subset_df)
modell3_mlr_alt.fit().summary()

# Henter intercept og slope for hver av temaene 'Star Wars' og 'NINJAGO'
intercept = [modell3_mlr.fit().params['Theme[T.Star Wars]'], modell3_mlr.fit().params['Theme[T.NINJAGO]'], 0] + modell3_mlr.fit().params['Intercept']
slope = modell3_mlr.fit().params['Pieces']  # Henter koeffisienten for 'Pieces' fra den multiple regresjonsmodellen

# Loop over hvert tema i 'mythemes'
for i, theme in enumerate(mythemes):  # 'mythemes' inneholder de temaene vi er interessert i (f.eks. 'Star Wars', 'NINJAGO', 'Harry Potter')

    # Filtrerer data for det aktuelle temaet og lagrer antall brikker (Pieces)
    regression_x = np.array(subset_df[subset_df['Theme'].isin([theme])]['Pieces'])  # Henter verdiene for 'Pieces' for det aktuelle temaet

    # Beregner predikert pris ved hjelp av den lineære regresjonsmodellen for det aktuelle temaet
    regression_y = slope * regression_x + intercept[i]  # Bruker den kalkulerte slope og intercept for å predikere 'Price'

    # Plotter scatter plot og regresjonslinje for hvert tema
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin([theme])]['Price'], color=plt.cm.tab10(i))  # Scatter plot for Pieces vs. Price
    plt.plot(regression_x, regression_y, color=plt.cm.tab10(i), label=theme)  # Plotter regresjonslinjen for det aktuelle temaet

# Plotter regresjonslinje uten tema som forklaringsvariabel
regression_x = np.array(subset_df['Pieces'])  # Henter 'Pieces' verdiene for hele datasetet
regression_y = res.sub.params['Pieces'] * regression_x + res.sub.params['Intercept']  # Beregner predikert 'Price' uten hensyn til 'Theme'
plt.plot(regression_x, regression_y, color='black', label='No theme')  # Plotter regresjonslinjen uten tema

# Legger til akseetiketter og tittel
plt.xlabel('Antall brikker')  # Setter etikett for x-aksen (Antall brikker)
plt.ylabel('Pris [$]')  # Setter etikett for y-aksen (Pris i dollar)
plt.title('Kryssplott med regresjonslinjer')  # Tittelen for plottet

# Viser legenden for å forklare farger og linjer
plt.legend()  # Viser legenden for scatter plottene og regresjonslinjene

# Legger til grid for bedre lesbarhet
plt.grid()  # Viser grid (rutenett) for å gjøre plottet mer lesbart

# Viser plottet
plt.show()  # Viser plottet på skjermen

# med interaksjonsledd mellom antall brikker og tema
modell3_mlri = smf.ols('Price ~ Pieces*Theme' , data = subset_df)
modell3_mlri.fit().summary()

# plott
intercept = [modell3_mlri.fit().params['Theme[T.Star Wars]'], modell3_mlri.fit().params['Theme[T.NINJAGO]'], 0] + modell3_mlri.fit().params['Intercept']
slope = [modell3_mlri.fit().params['Pieces:Theme[T.Star Wars]'], modell3_mlri.fit().params['Pieces:Theme[T.NINJAGO]'], 0] + modell3_mlri.fit().params['Pieces']

for i, theme in enumerate(mythemes):

    regression_x = np.array(subset_df[subset_df['Theme'].isin([theme])]['Pieces'])
    regression_y = slope[i] * regression_x + intercept[i]

    # Plot scatter plot and regression line
    plt.scatter(subset_df[subset_df['Theme'].isin([theme])]['Pieces'], subset_df[subset_df['Theme'].isin([theme])]['Price'], color=plt.cm.tab10(i))
    plt.plot(regression_x, regression_y, color=plt.cm.tab10(i), label=theme)
    
# uten tema som forklaringsvariabel:
regression_x = np.array(subset_df['Pieces'])
regression_y = res_sub.params['Pieces'] * regression_x + res_sub.params['Intercept']
plt.plot(regression_x, regression_y, color='black', label='Theme unaccounted for')
    
plt.xlabel('Antall brikker')
plt.ylabel('Pris [$]')
plt.title('Kryssplott med regresjonslinjer')
plt.legend()
plt.grid()
plt.show()

# Steg 5: Evaluere om modellen passer til dataene
# Plotte predikert verdi mot residual
figure, axis = plt.subplots(1, 2, figsize = (15, 5))
sns.scatterplot(x = modell3_mlri.fit().fittedvalues, y = modell3_mlri.fit().resid, ax = axis[0])
axis[0].set_ylabel("Residual")
axis[0].set_xlabel("Predikert verdi")

# Lage kvantil-kvantil-plott for residualene
sm.qqplot(modell3_mlri.fit().resid, line = '45', fit = True, ax = axis[1])
axis[1].set_ylabel("Kvantiler i residualene")
axis[1].set_xlabel("Kvantiler i normalfordelingen")
plt.show()

# Gruppere temaer i nye grupper:
# (Harry Potter, NINJAGO og Star Wars havner i én gruppe, City og Friends i en annen, og alle andre i en tredje)
df2['cat'] = np.where(df2['Theme'].isin(['Harry Potter', 'NINJAGO', 'Star Wars']), 'Cat1', 
                      np.where(df2['Theme'].isin(['City', 'Friends']), 'Cat2', 'Cat3'))
df2.groupby(['cat']).size().reset_index(name = 'Count')

df2.groupby(['cat', 'Theme']).size().reset_index(name = 'Count')
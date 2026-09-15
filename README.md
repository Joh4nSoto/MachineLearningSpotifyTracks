# Análisis de Popularidad de Canciones en Spotify

**Autores:** Johan Soto, Fernanda Alcaino

---

## **CONTEXTO DE LOS DATOS**

- **track_id**: El ID de Spotify para la pista.
- **artists**: Los nombres de los artistas que interpretaron la pista. Si hay más de un artista, se separan con un `;`.
- **album_name**: El nombre del álbum en el que aparece la pista.
- **track_name**: Nombre de la pista.
- **popularity**: La popularidad de una canción es un valor entre 0 y 100, donde 100 representa la máxima popularidad. Se calcula mediante un algoritmo y se basa, principalmente, en el número total de reproducciones y la antigüedad de estas. En general, las canciones que se reproducen mucho actualmente tendrán mayor popularidad que las que se reprodujeron mucho en el pasado. Las canciones duplicadas (por ejemplo, la misma canción de un sencillo y de un álbum) se valoran de forma independiente. La popularidad del artista y del álbum se deriva matemáticamente de la popularidad de la canción.
- **duration_ms**: La duración de la pista en milisegundos.
- **explicit**: Indica si la canción contiene letras explícitas (verdadero = sí; falso = no; desconocido).
- **danceability**: La bailabilidad describe qué tan adecuada es una pista para bailar basándose en una combinación de elementos musicales que incluyen el tempo, la estabilidad del ritmo, la fuerza del compás y la regularidad general. Un valor de 0,0 es la menos bailable y 1,0 es la más bailable.
- **energy**: La energía es una medida de 0,0 a 1,0 y representa una medida perceptual de intensidad y actividad. Por lo general, las pistas enérgicas se sienten rápidas, fuertes y ruidosas. Por ejemplo, el death metal tiene alta energía, mientras que un preludio de Bach obtiene una puntuación baja en la escala.
- **key**: La clave en la que se encuentra la pista. Los números enteros se corresponden con los tonos utilizando la notación estándar de la clase de tono. Por ejemplo 0 = C, 1 = C♯/D♭, 2 = D, etc. Si no se detectó ninguna clave, el valor es -1.
- **loudness**: El nivel de sonoridad general de una pista en decibelios (dB).
- **mode**: El modo indica la modalidad (mayor o menor) de una pista, el tipo de escala de la que se deriva su contenido melódico. La modalidad mayor se representa con 1 y la menor con 0.
- **speechiness**: La detección de palabras habladas en una pista indica que la grabación es más exclusivamente hablada (por ejemplo, un programa de entrevistas, un audiolibro, poesía), más cerca de 1,0 estará el valor del atributo. Los valores superiores a 0,66 describen pistas que probablemente estén compuestas enteramente de palabras habladas. Los valores entre 0,33 y 0,66 describen pistas que pueden contener tanto música como habla, ya sea en secciones o superpuestas, incluyendo casos como la música rap. Los valores inferiores a 0,33 probablemente representan música y otras pistas que no son habladas.
- **acousticness**: Una medida de confianza de 0,0 a 1,0 sobre si la pista es acústica. 1,0 representa una alta confianza de que la pista es acústica.
- **instrumentalness**: Predice si una pista no contiene voces. Los sonidos "ooh" y "aah" se tratan como instrumentales en este contexto. Las pistas de rap o palabra hablada son claramente "vocales". Cuanto más cerca esté el valor de instrumentalidad de 1.0, mayor será la probabilidad de que la pista no contenga contenido vocal.
- **liveness**: Detecta la presencia de público en la grabación. Valores de Liveness más altos representan una mayor probabilidad de que la pista se haya interpretado en directo. Un valor superior a 0,8 indica una alta probabilidad de que la pista sea en directo.
- **valence**: Una medida de 0,0 a 1,0 que describe la positividad musical que transmite una pista. Las pistas con alta valencia suenan más positivas (por ejemplo, alegres, joviales, eufóricas), mientras que las pistas con baja valencia suenan más negativas (por ejemplo, tristes, deprimidas, enojadas).
- **tempo**: El tempo estimado general de una pista en pulsaciones por minuto (BPM). En terminología musical, el tempo es la velocidad o ritmo de una pieza determinada y se deriva directamente de la duración promedio de las pulsaciones.
- **time_signature**: Un compás estimado. El compás (o métrica) es una convención de notación para especificar cuántos pulsos hay en cada compás (o medida). El compás varía de 3 a 7, indicando compases de 3/4, a 7/4.
- **track_genre**: El género al que pertenece la pista.

---

## **DESCRIPCIÓN DEL PROBLEMA DE NEGOCIO**

La industria musical actual es un mercado altamente saturado y competitivo, donde miles de canciones son publicadas diariamente en plataformas de streaming como Spotify. En este contexto, productores, artistas y sellos discográficos invierten grandes cantidades de recursos en la creación y comercialización de canciones sin tener certeza de su éxito comercial. Existe una alta incertidumbre respecto a qué combinaciones específicas de características sonoras (como la energía, qué tan bailable es la canción o el tono) capturan mejor la atención de los oyentes.

Para resolver este problema, se busca desarrollar una inteligencia musical basada en algoritmos predictivos que estimen la popularidad de las pistas. Esto permitirá anticipar el éxito comercial, tomar decisiones estratégicas más precisas y evitar el gasto ineficiente de presupuesto promocional en canciones que probablemente no tendrán un alto impacto.

---

## **OBJETIVOS DEL PROYECTO**

Desarrollar un modelo predictivo de Machine Learning que permita estimar la popularidad de las canciones en Spotify a partir de sus características sonoras y datos, con el fin de optimizar la toma de decisiones comerciales y la asignación de presupuestos en la producción y promoción musical.

---

## **DEFINICIÓN DE KPIs**

Los KPIs permitirán medir el comportamiento de las canciones y facilitar la identificación de patrones asociados a su popularidad. Estos indicadores estarán orientados principalmente a comprender el comportamiento del catálogo musical y apoyar la toma de decisiones.

---

## **1. CARGA INICIAL EDA**

```python
import pandas as pd
df = pd.read_csv("/content/Spotify_Tracks_Dataset.csv")
```

```python
df.head()
```

**Salida:**

| | Unnamed: 0 | track_id | artists | album_name | track_name | popularity | duration_ms | explicit | danceability | energy | ... | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo | time_signature | track_genre |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 5SuOikwiRyPMVoIQDJUgSV | Gen Hoshino | Comedy | Comedy | 73 | 230666 | False | 0.676 | 0.4610 | ... | -6.746 | 0 | 0.1430 | 0.0322 | 0.000001 | 0.3580 | 0.715 | 87.917 | 4 | acoustic |
| 1 | 1 | 4qPNDBW1i3p13qLCt0Ki3A | Ben Woodward | Ghost (Acoustic) | Ghost - Acoustic | 55 | 149610 | False | 0.420 | 0.1660 | ... | -17.235 | 1 | 0.0763 | 0.9240 | 0.000006 | 0.1010 | 0.267 | 77.489 | 4 | acoustic |
| 2 | 2 | 1iJBSr7s7jYXzM8EGcbK5b | Ingrid Michaelson;ZAYN | To Begin Again | To Begin Again | 57 | 210826 | False | 0.438 | 0.3590 | ... | -9.734 | 1 | 0.0557 | 0.2100 | 0.000000 | 0.1170 | 0.120 | 76.332 | 4 | acoustic |
| 3 | 3 | 6lfxq3CG4xtTiEg7opyCyx | Kina Grannis | Crazy Rich Asians (Original Motion Picture Sou... | Can't Help Falling In Love | 71 | 201933 | False | 0.266 | 0.0596 | ... | -18.515 | 1 | 0.0363 | 0.9050 | 0.000071 | 0.1320 | 0.143 | 181.740 | 3 | acoustic |
| 4 | 4 | 5vjLSffimiIP26QG5WcN2K | Chord Overstreet | Hold On | Hold On | 82 | 198853 | False | 0.618 | 0.4430 | ... | -9.681 | 1 | 0.0526 | 0.4690 | 0.000000 | 0.0829 | 0.167 | 119.949 | 4 | acoustic |

*(5 filas × 21 columnas)*

```python
df.info()
```

**Salida:**

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 114000 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        114000 non-null  int64  
 1   track_id          114000 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        114000 non-null  int64  
 6   duration_ms       114000 non-null  int64  
 7   explicit          114000 non-null  bool   
 8   danceability      114000 non-null  float64
 9   energy            114000 non-null  float64
 10  key               114000 non-null  int64  
 11  loudness          114000 non-null  float64
 12  mode              114000 non-null  int64  
 13  speechiness       114000 non-null  float64
 14  acousticness      114000 non-null  float64
 15  instrumentalness  114000 non-null  float64
 16  liveness          114000 non-null  float64
 17  valence           114000 non-null  float64
 18  tempo             114000 non-null  float64
 19  time_signature    114000 non-null  int64  
 20  track_genre       114000 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
memory usage: 17.5+ MB
```

---

## **2. BÚSQUEDA DE NULOS Y DUPLICADOS**

```python
df.isnull().sum()
```

**Salida:**

| Columna | Nulos |
|---|---|
| Unnamed: 0 | 0 |
| track_id | 0 |
| artists | 1 |
| album_name | 1 |
| track_name | 1 |
| popularity | 0 |
| duration_ms | 0 |
| explicit | 0 |
| danceability | 0 |
| energy | 0 |
| key | 0 |
| loudness | 0 |
| mode | 0 |
| speechiness | 0 |
| acousticness | 0 |
| instrumentalness | 0 |
| liveness | 0 |
| valence | 0 |
| tempo | 0 |
| time_signature | 0 |
| track_genre | 0 |

> Solo vienen 3 datos con valores nulos, así que los visualizamos.

```python
df[df['artists'].isnull()]
```

**Salida:**

| | Unnamed: 0 | track_id | artists | album_name | track_name | popularity | duration_ms | explicit | danceability | energy | ... | loudness | mode | speechiness | acousticness | instrumentalness | liveness | valence | tempo | time_signature | track_genre |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 65900 | 65900 | 1kR4gIb7nGxHPI3D2ifs59 | NaN | NaN | NaN | 0 | 0 | False | 0.501 | 0.583 | ... | -9.46 | 0 | 0.0605 | 0.69 | 0.00396 | 0.0747 | 0.734 | 138.391 | 4 | k-pop |

> Y nos dimos cuenta de que los únicos 3 valores nulos corresponden a la misma canción, con popularidad 0, así que no nos afecta en lo absoluto eliminarla.

```python
df.drop(65900, inplace=True)
df.info()
```

**Salida:**

```
<class 'pandas.core.frame.DataFrame'>
Index: 113999 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        113999 non-null  int64  
 1   track_id          113999 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        113999 non-null  int64  
 6   duration_ms       113999 non-null  int64  
 7   explicit          113999 non-null  bool   
 8   danceability      113999 non-null  float64
 9   energy            113999 non-null  float64
 10  key               113999 non-null  int64  
 11  loudness          113999 non-null  float64
 12  mode              113999 non-null  int64  
 13  speechiness       113999 non-null  float64
 14  acousticness      113999 non-null  float64
 15  instrumentalness  113999 non-null  float64
 16  liveness          113999 non-null  float64
 17  valence           113999 non-null  float64
 18  tempo             113999 non-null  float64
 19  time_signature    113999 non-null  int64  
 20  track_genre       113999 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
memory usage: 18.4+ MB
```

```python
df.isnull().sum()
```

**Salida:**

*(Todos los valores son 0)*

```python
df.duplicated().sum()
```

**Salida:** `np.int64(0)`

> Viene sin duplicados.

---

## **3. VISUALIZACIÓN VARIABLE POR VARIABLE**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_seleccion = df[["artists", "album_name","track_name","popularity",
                   "duration_ms","explicit","danceability",
                   "energy", "loudness", "key", "mode","speechiness",
                   "acousticness","instrumentalness","liveness",
                   "valence","tempo","time_signature","track_genre"]]

# Iteramos sobre el DataFrame seleccionado
for col in df_seleccion.columns:
    # IMPORTANTE: Validar tipo usando el DataFrame seleccionado
    tipo = df_seleccion[col].dtype

    plt.figure(figsize=(7, 4))

    # CASO 1: int64 u object -> Gráfico de barras (conteo de frecuencias)
    if tipo == 'int64' or tipo == 'object':
        # SOLUCIÓN: Si hay demasiados valores únicos, graficamos solo el Top 10
        top_valores = df_seleccion[col].value_counts().head(10)

        # Usamos los datos filtrados del Top 10 para que cargue instantáneamente
        sns.countplot(
            x=df_seleccion[df_seleccion[col].isin(top_valores.index)][col],
            order=top_valores.index,
            palette='viridis'
        )

        # Rotamos las etiquetas del eje X para que los nombres largos no se encimen
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top 10 Frecuencias de {col} ({tipo}) - Barras')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')

    # CASO 2: float64 -> Histograma (distribución de valores continuos)
    elif tipo == 'float64':
        sns.histplot(df_seleccion[col].dropna(), kde=True, color='skyblue', bins=20)
        plt.title(f'Distribución de {col} ({tipo}) - Histograma')
        plt.xlabel(col)
        plt.ylabel('Frecuencia')

    # CASO 3: bool -> Gráfico de tarta (proporción de True/False)
    elif tipo == 'bool':
        conteo = df_seleccion[col].value_counts()
        plt.pie(conteo, labels=conteo.index, autopct='%1.1f%%', colors=['#4CAF50', '#FF5722'], startangle=90)
        plt.title(f'Proporción de {col} ({tipo}) - Tarta')

    plt.tight_layout()
    plt.show()
```

**Salida (resumida):**

Se generaron gráficos individuales para cada variable:

- **Variables categóricas / enteras (Top 10 frecuencias):** `artists`, `album_name`, `track_name`, `popularity`, `duration_ms`, `key`, `mode`, `time_signature`, `track_genre`.
- **Variables continuas (Histogramas):** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`.
- **Variable booleana (Tarta):** `explicit`.

**Observaciones destacadas:**

- La popularidad se distribuye mayormente entre 0 y 80, con un pico importante en 0 (canciones sin popularidad registrada).
- Los géneros más frecuentes en el Top 10 son: `acoustic`, `afrobeat`, `alt-rock`, `alternative`, `ambient`, `anime`, `black-metal`, `bluegrass`, `blues`, `brazil`.
- La duración de las pistas se concentra entre 150,000 y 250,000 ms.
- La mayoría de las canciones no son explícitas.
- El compás más común es 4/4 (`time_signature` = 4).
- La tonalidad más común es 0 (C), 1 (C♯/D♭), y 2 (D).

---

## **4. COMPARACIÓN DE VARIABLES CON MAPA DE CALOR**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Filtrar solo las columnas numéricas de tu selección
df_numerico = df_seleccion.select_dtypes(include=[np.number])

# 2. Calcular la matriz de correlación
matriz_correlacion = df_numerico.corr()

# 3. Configurar el tamaño del gráfico
plt.figure(figsize=(12, 10))

# 4. Dibujar el mapa de calor
sns.heatmap(
    matriz_correlacion,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    linewidths=0.5
)

# 5. Personalizar títulos
plt.title('Mapa de Calor: Correlación de Variables Numéricas con Popularidad', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
```

**Salida:**

![Mapa de Calor - Correlación de Variables Numéricas con Popularidad](heatmap.png)

**Hallazgos del mapa de calor:**

- **Popularidad vs. Loudness:** correlación negativa moderada (~ -0.02), prácticamente nula.
- **Energy vs. Loudness:** correlación positiva fuerte (~ 0.76), lo cual es esperable (canciones más enérgicas suelen ser más ruidosas).
- **Energy vs. Acousticness:** correlación negativa fuerte (~ -0.73), las canciones acústicas tienden a tener baja energía.
- **Danceability vs. Valence:** correlación positiva moderada (~ 0.47).
- **Popularity** presenta correlaciones débiles con todas las variables numéricas (< 0.1 en valor absoluto), lo que sugiere que la popularidad es difícil de predecir linealmente a partir de estas características.

---

## **5. VISUALIZACIÓN**

### 5.1 Top 15 Géneros con Mayor Popularidad Promedio

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calcular la popularidad promedio por género y seleccionar los 15 más altos
top_generos_populares = (
    df_seleccion.groupby('track_genre')['popularity']
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .index
)

# 2. Filtrar el DataFrame para quedarnos solo con esos 15 mejores géneros
df_top_generos = df_seleccion[df_seleccion['track_genre'].isin(top_generos_populares)]

# 3. Configurar el lienzo del gráfico
plt.figure(figsize=(12, 6))

# 4. Crear el gráfico de barras (ordenado de mayor a menor popularidad promedio)
sns.barplot(
    x='popularity',
    y='track_genre',
    data=df_top_generos,
    order=top_generos_populares,
    palette='viridis',
    errorbar=None
)

# 5. Personalizar el gráfico
plt.title('Top 15 Géneros Musicales con Mayor Popularidad Promedio', fontsize=16, pad=15)
plt.xlabel('Popularidad Promedio', fontsize=12)
plt.ylabel('Género de la Canción', fontsize=12)

plt.tight_layout()
plt.show()
```

**Salida:**

![Top 15 Géneros con Mayor Popularidad Promedio](top_generos_populares.png)

**Conclusión:** El género de música más popular es **pop-film**, seguido de **k-pop** y **chill**.

---

### 5.2 Top 15 Géneros con Menor Popularidad Promedio

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calcular la popularidad promedio por género y seleccionar los 15 MÁS BAJOS
top_generos_menos_populares = (
    df_seleccion.groupby('track_genre')['popularity']
    .mean()
    .sort_values(ascending=True)
    .head(15)
    .index
)

# 2. Filtrar el DataFrame para quedarnos solo con esos 15 géneros
df_menos_populares = df_seleccion[df_seleccion['track_genre'].isin(top_generos_menos_populares)]

# 3. Configurar el lienzo del gráfico
plt.figure(figsize=(12, 6))

# 4. Crear el gráfico de barras (ordenado de menor a mayor popularidad promedio)
sns.barplot(
    x='popularity',
    y='track_genre',
    data=df_menos_populares,
    order=top_generos_menos_populares,
    palette='magma',
    errorbar=None
)

# 5. Personalizar el gráfico
plt.title('Top 15 Géneros Musicales con MENOR Popularidad Promedio', fontsize=16, pad=15)
plt.xlabel('Popularidad Promedio', fontsize=12)
plt.ylabel('Género de la Canción', fontsize=12)

plt.tight_layout()
plt.show()
```

**Salida:**

![Top 15 Géneros con Menor Popularidad Promedio](top_generos_menos_populares.png)

**Conclusión:** Los géneros musicales menos populares son **iranian**, seguido de **romance** y **latin**.

---

### 5.3 Popularidad Promedio según el Tempo (BPM)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Crear intervalos de tempo de 10 en 10 BPM (ej. de 60 a 200+ BPM)
bins = range(60, 210, 10)
df_seleccion['tempo_rango'] = pd.cut(df_seleccion['tempo'], bins=bins)

# 2. Calcular la popularidad promedio para cada rango de tempo
tempo_pop = (
    df_seleccion.groupby('tempo_rango', observed=False)['popularity']
    .mean()
    .reset_index()
)

# 3. Configurar el gráfico
plt.figure(figsize=(12, 6))

# 4. Crear un gráfico de líneas con puntos para ver la tendencia claramente
sns.lineplot(
    x=tempo_pop['tempo_rango'].astype(str),
    y=tempo_pop['popularity'],
    marker='o',
    linewidth=2.5,
    color='#1DB954'
)

# 5. Personalizar el diseño
plt.title('Popularidad Promedio de las Canciones según su Tempo (BPM)', fontsize=16, pad=15)
plt.xlabel('Rango de Tempo (Beats Per Minute)', fontsize=12)
plt.ylabel('Popularidad Promedio', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

**Salida:**

![Popularidad Promedio según Tempo](tempo_popularidad.png)

**Conclusión:** El rango óptimo de tempo para una popularidad media está entre los **80 y 170 BPM**, con un mayor alcance de popularidad en los **140 BPM**.

---

### 5.4 Popularidad Promedio según el Compás (Time Signature)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calcular la popularidad promedio para cada compás musical
time_pop = (
    df_seleccion.groupby('time_signature')['popularity']
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

# 2. Configurar el lienzo del gráfico
plt.figure(figsize=(10, 5))

# 3. Crear el gráfico de barras
sns.barplot(
    x='time_signature',
    y='popularity',
    data=time_pop,
    order=time_pop['time_signature'],
    palette='Blues_r',
    errorbar=None
)

# 4. Personalizar el diseño
plt.title('Popularidad Promedio de las Canciones según su Compás (Time Signature)', fontsize=16, pad=15)
plt.xlabel('Signatura de Tiempo (Compás)', fontsize=12)
plt.ylabel('Popularidad Promedio', fontsize=12)

# Añadir etiquetas de valor sobre cada barra
for index, row in time_pop.iterrows():
    plt.text(
        index,
        row['popularity'] + 1,
        f"{row['popularity']:.1f}",
        color='black',
        ha="center",
        fontsize=10
    )

plt.tight_layout()
plt.show()
```

**Salida:**

![Popularidad Promedio según Compás](time_signature_popularidad.png)

**Conclusión:** El tiempo de compás más popular (descartando el tempo = 0, que es imposible y por lo tanto dato inválido) es el de **4/4**.

---

### 5.5 Perfil Acústico del Top de Canciones Populares

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Definir qué es una canción "popular" (popularidad >= 75)
top_canciones = df_seleccion[df_seleccion['popularity'] >= 75]

# 2. Seleccionar las variables acústicas que están en la misma escala (0 a 1)
variables_acusticas = [
    'danceability', 'energy', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence'
]

# 3. Calcular el promedio de cada parámetro para este grupo exitoso
promedios_top = top_canciones[variables_acusticas].mean().sort_values(ascending=False)

# 4. Configurar el gráfico
plt.figure(figsize=(10, 6))

# 5. Crear el gráfico de barras horizontales
sns.barplot(
    x=promedios_top.values,
    y=promedios_top.index,
    palette='viridis',
    errorbar=None
)

# 6. Personalizar el diseño
plt.title('Perfil Acústico del Top de Canciones Populares (Escala 0 a 1)', fontsize=15, pad=15)
plt.xlabel('Valor Promedio (Cercano a 1 es Mayor Presencia)', fontsize=12)
plt.ylabel('Parámetro Acústico', fontsize=12)
plt.xlim(0, 1)

# Añadir los valores exactos al final de cada barra
for index, value in enumerate(promedios_top.values):
    plt.text(value + 0.02, index, f"{value:.2f}", va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()
```

**Salida:**

![Perfil Acústico del Top de Canciones Populares](perfil_acustico.png)

**Conclusión:** Las canciones más populares (popularidad ≥ 75) tienden a tener:

- **Danceability** alto (~0.65)
- **Energy** moderada-alta (~0.64)
- **Valence** moderada (~0.45)
- **Liveness** bajo (~0.20)
- **Speechiness** bajo (~0.08)
- **Acousticness** bajo (~0.25)
- **Instrumentalness** muy bajo (~0.05)

Esto sugiere que las canciones exitosas son bailables, con energía moderada-alta, positivas (valence), no acústicas, con voces (baja instrumentalidad) y no son en directo.

---

## **CONCLUSIONES GENERALES**

1. **Datos limpios:** El dataset contiene 113,999 canciones después de eliminar la única fila con valores nulos (que además tenía popularidad 0). No hay duplicados.

2. **Popularidad difícil de predecir linealmente:** Las correlaciones entre `popularity` y las variables numéricas son muy débiles (< 0.1 en valor absoluto), lo que indica que la popularidad depende de múltiples factores no lineales y probablemente de variables externas (marketing, artista, momento de lanzamiento, etc.).

3. **Géneros más populares:** `pop-film`, `k-pop` y `chill` lideran la popularidad promedio.

4. **Géneros menos populares:** `iranian`, `romance` y `latin` tienen las popularidades promedio más bajas.

5. **Tempo óptimo:** Canciones entre 80 y 170 BPM tienden a tener mejor popularidad, con un pico en 140 BPM.

6. **Compás dominante:** El compás 4/4 es el más común y el más popular.

7. **Perfil de canción exitosa:** Las canciones con popularidad ≥ 75 son bailables, energéticas, positivas, no acústicas, con voces y no en directo.

---

*Documento generado a partir del notebook `ev1_ML_JohanSoto_FernandaAlcaino.ipynb`*

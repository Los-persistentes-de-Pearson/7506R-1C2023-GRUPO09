---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Importamos
```python
try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold,RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix, classification_report , f1_score
from sklearn.tree import DecisionTreeClassifier

#Si estamos  en colab tenemos que instalar la libreria "dtreeviz" aparte. 
if IN_COLAB == True:
    !pip install 'dtreeviz'
import dtreeviz.trees as dtreeviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
```

```python
#Diccionarios tomados de la libreria pycountry-convert: https://github.com/jefftune/pycountry-convert
COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2 = {
    'ABH': 'AB',
    'ABW': 'AW',
    'AFG': 'AF',
    'AGO': 'AO',
    'AIA': 'AI',
    'ALA': 'AX',
    'ALB': 'AL',
    'AND': 'AD',
    'ARE': 'AE',
    'ARG': 'AR',
    'ARM': 'AM',
    'ASM': 'AS',
    'ATG': 'AG',
    'AUS': 'AU',
    'AUT': 'AT',
    'AZE': 'AZ',
    'BDI': 'BI',
    'BEL': 'BE',
    'BEN': 'BJ',
    'BFA': 'BF',
    'BGD': 'BD',
    'BGR': 'BG',
    'BHR': 'BH',
    'BHS': 'BS',
    'BIH': 'BA',
    'BLM': 'BL',
    'BLR': 'BY',
    'BLZ': 'BZ',
    'BMU': 'BM',
    'BOL': 'BO',
    'BRA': 'BR',
    'BRB': 'BB',
    'BRN': 'BN',
    'BTN': 'BT',
    'BVT': 'BV',
    'BWA': 'BW',
    'CAF': 'CF',
    'CAN': 'CA',
    'CCK': 'CC',
    'CHE': 'CH',
    'CHL': 'CL',
    'CHN': 'CN',
    'CIV': 'CI',
    'CMR': 'CM',
    'COD': 'CD',
    'COG': 'CG',
    'COK': 'CK',
    'COL': 'CO',
    'COM': 'KM',
    'CPV': 'CV',
    'CRI': 'CR',
    'CUB': 'CU',
    'CUW': 'CW',
    'CXR': 'CX',
    'CYM': 'KY',
    'CYP': 'CY',
    'CZE': 'CZ',
    'DEU': 'DE',
    'DJI': 'DJ',
    'DMA': 'DM',
    'DNK': 'DK',
    'DOM': 'DO',
    'DZA': 'DZ',
    'ECU': 'EC',
    'EGY': 'EG',
    'ERI': 'ER',
    'ESP': 'ES',
    'EST': 'EE',
    'ETH': 'ET',
    'FIN': 'FI',
    'FJI': 'FJ',
    'FLK': 'FK',
    'FRA': 'FR',
    'FRO': 'FO',
    'FSM': 'FM',
    'GAB': 'GA',
    'GBR': 'GB',
    'GEO': 'GE',
    'GGY': 'GG',
    'GHA': 'GH',
    'GIB': 'GI',
    'GIN': 'GN',
    'GLP': 'GP',
    'GMB': 'GM',
    'GNB': 'GW',
    'GNQ': 'GQ',
    'GRC': 'GR',
    'GRD': 'GD',
    'GRL': 'GL',
    'GTM': 'GT',
    'GUF': 'GF',
    'GUM': 'GU',
    'GUY': 'GY',
    'HKG': 'HK',
    'HMD': 'HM',
    'HND': 'HN',
    'HRV': 'HR',
    'HTI': 'HT',
    'HUN': 'HU',
    'IDN': 'ID',
    'IMN': 'IM',
    'IND': 'IN',
    'IOT': 'IO',
    'IRL': 'IE',
    'IRN': 'IR',
    'IRQ': 'IQ',
    'ISL': 'IS',
    'ISR': 'IL',
    'ITA': 'IT',
    'JAM': 'JM',
    'JEY': 'JE',
    'JOR': 'JO',
    'JPN': 'JP',
    'KAZ': 'KZ',
    'KEN': 'KE',
    'KGZ': 'KG',
    'KHM': 'KH',
    'KIR': 'KI',
    'KNA': 'KN',
    'KOR': 'KR',
    'KWT': 'KW',
    'LAO': 'LA',
    'LBN': 'LB',
    'LBR': 'LR',
    'LBY': 'LY',
    'LCA': 'LC',
    'LIE': 'LI',
    'LKA': 'LK',
    'LSO': 'LS',
    'LTU': 'LT',
    'LUX': 'LU',
    'LVA': 'LV',
    'MAC': 'MO',
    'MAF': 'MF',
    'MAR': 'MA',
    'MCO': 'MC',
    'MDA': 'MD',
    'MDG': 'MG',
    'MDV': 'MV',
    'MEX': 'MX',
    'MHL': 'MH',
    'MKD': 'MK',
    'MLI': 'ML',
    'MLT': 'MT',
    'MMR': 'MM',
    'MNE': 'ME',
    'MNG': 'MN',
    'MNP': 'MP',
    'MOZ': 'MZ',
    'MRT': 'MR',
    'MSR': 'MS',
    'MTQ': 'MQ',
    'MUS': 'MU',
    'MWI': 'MW',
    'MYS': 'MY',
    'MYT': 'YT',
    'NAM': 'NA',
    'NCL': 'NC',
    'NER': 'NE',
    'NFK': 'NF',
    'NGA': 'NG',
    'NIC': 'NI',
    'NIU': 'NU',
    'NLD': 'NL',
    'NOR': 'NO',
    'NPL': 'NP',
    'NRU': 'NR',
    'NZL': 'NZ',
    'OMN': 'OM',
    'OST': 'OS',
    'PAK': 'PK',
    'PAN': 'PA',
    'PER': 'PE',
    'PHL': 'PH',
    'PLW': 'PW',
    'PNG': 'PG',
    'POL': 'PL',
    'PRI': 'PR',
    'PRK': 'KP',
    'PRT': 'PT',
    'PRY': 'PY',
    'PSE': 'PS',
    'PYF': 'PF',
    'QAT': 'QA',
    'REU': 'RE',
    'ROU': 'RO',
    'RUS': 'RU',
    'RWA': 'RW',
    'SAU': 'SA',
    'SDN': 'SD',
    'SEN': 'SN',
    'SGP': 'SG',
    'SGS': 'GS',
    'SHN': 'SH',
    'SLB': 'SB',
    'SLE': 'SL',
    'SLV': 'SV',
    'SMR': 'SM',
    'SOM': 'SO',
    'SPM': 'PM',
    'SRB': 'RS',
    'SSD': 'SS',
    'STP': 'ST',
    'SUR': 'SR',
    'SVK': 'SK',
    'SVN': 'SI',
    'SWE': 'SE',
    'SWZ': 'SZ',
    'SYC': 'SC',
    'SYR': 'SY',
    'TCA': 'TC',
    'TCD': 'TD',
    'TGO': 'TG',
    'THA': 'TH',
    'TJK': 'TJ',
    'TKL': 'TK',
    'TKM': 'TM',
    'TON': 'TO',
    'TMP': 'TP',
    'TTO': 'TT',
    'TUN': 'TN',
    'TUR': 'TR',
    'TUV': 'TV',
    'TWN': 'TW',
    'TZA': 'TZ',
    'UAE': 'AE',
    'UGA': 'UG',
    'UKR': 'UA',
    'URY': 'UY',
    'USA': 'US',
    'UZB': 'UZ',
    'VCT': 'VC',
    'VEN': 'VE',
    'VGB': 'VG',
    'VIR': 'VI',
    'VNM': 'VN',
    'VUT': 'VU',
    'WLF': 'WF',
    'WSM': 'WS',
    'YEM': 'YE',
    'ZAF': 'ZA',
    'ZMB': 'ZM',
    'ZWE': 'ZW',
}

COUNTRY_ALPHA2_TO_CONTINENT = {
    'AB': 'Asia',
    'AD': 'Europe',
    'AE': 'Asia',
    'AF': 'Asia',
    'AG': 'North America',
    'AI': 'North America',
    'AL': 'Europe',
    'AM': 'Asia',
    'AO': 'Africa',
    'AR': 'South America',
    'AS': 'Oceania',
    'AT': 'Europe',
    'AU': 'Oceania',
    'AW': 'North America',
    'AX': 'Europe',
    'AZ': 'Asia',
    'BA': 'Europe',
    'BB': 'North America',
    'BD': 'Asia',
    'BE': 'Europe',
    'BF': 'Africa',
    'BG': 'Europe',
    'BH': 'Asia',
    'BI': 'Africa',
    'BJ': 'Africa',
    'BL': 'North America',
    'BM': 'North America',
    'BN': 'Asia',
    'BO': 'South America',
    'BQ': 'North America',
    'BR': 'South America',
    'BS': 'North America',
    'BT': 'Asia',
    'BV': 'Antarctica',
    'BW': 'Africa',
    'BY': 'Europe',
    'BZ': 'North America',
    'CA': 'North America',
    'CC': 'Asia',
    'CD': 'Africa',
    'CF': 'Africa',
    'CG': 'Africa',
    'CH': 'Europe',
    'CI': 'Africa',
    'CK': 'Oceania',
    'CL': 'South America',
    'CM': 'Africa',
    'CN': 'Asia',
    'CO': 'South America',
    'CR': 'North America',
    'CU': 'North America',
    'CV': 'Africa',
    'CW': 'North America',
    'CX': 'Asia',
    'CY': 'Asia',
    'CZ': 'Europe',
    'DE': 'Europe',
    'DJ': 'Africa',
    'DK': 'Europe',
    'DM': 'North America',
    'DO': 'North America',
    'DZ': 'Africa',
    'EC': 'South America',
    'EE': 'Europe',
    'EG': 'Africa',
    'ER': 'Africa',
    'ES': 'Europe',
    'ET': 'Africa',
    'FI': 'Europe',
    'FJ': 'Oceania',
    'FK': 'South America',
    'FM': 'Oceania',
    'FO': 'Europe',
    'FR': 'Europe',
    'GA': 'Africa',
    'GB': 'Europe',
    'GD': 'North America',
    'GE': 'Asia',
    'GF': 'South America',
    'GG': 'Europe',
    'GH': 'Africa',
    'GI': 'Europe',
    'GL': 'North America',
    'GM': 'Africa',
    'GN': 'Africa',
    'GP': 'North America',
    'GQ': 'Africa',
    'GR': 'Europe',
    'GS': 'South America',
    'GT': 'North America',
    'GU': 'Oceania',
    'GW': 'Africa',
    'GY': 'South America',
    'HK': 'Asia',
    'HM': 'Antarctica',
    'HN': 'North America',
    'HR': 'Europe',
    'HT': 'North America',
    'HU': 'Europe',
    'ID': 'Asia',
    'IE': 'Europe',
    'IL': 'Asia',
    'IM': 'Europe',
    'IN': 'Asia',
    'IO': 'Asia',
    'IQ': 'Asia',
    'IR': 'Asia',
    'IS': 'Europe',
    'IT': 'Europe',
    'JE': 'Europe',
    'JM': 'North America',
    'JO': 'Asia',
    'JP': 'Asia',
    'KE': 'Africa',
    'KG': 'Asia',
    'KH': 'Asia',
    'KI': 'Oceania',
    'KM': 'Africa',
    'KN': 'North America',
    'KP': 'Asia',
    'KR': 'Asia',
    'KW': 'Asia',
    'KY': 'North America',
    'KZ': 'Asia',
    'LA': 'Asia',
    'LB': 'Asia',
    'LC': 'North America',
    'LI': 'Europe',
    'LK': 'Asia',
    'LR': 'Africa',
    'LS': 'Africa',
    'LT': 'Europe',
    'LU': 'Europe',
    'LV': 'Europe',
    'LY': 'Africa',
    'MA': 'Africa',
    'MC': 'Europe',
    'MD': 'Europe',
    'ME': 'Europe',
    'MF': 'North America',
    'MG': 'Africa',
    'MH': 'Oceania',
    'MK': 'Europe',
    'ML': 'Africa',
    'MM': 'Asia',
    'MN': 'Asia',
    'MO': 'Asia',
    'MP': 'Oceania',
    'MQ': 'North America',
    'MR': 'Africa',
    'MS': 'North America',
    'MT': 'Europe',
    'MU': 'Africa',
    'MV': 'Asia',
    'MW': 'Africa',
    'MX': 'North America',
    'MY': 'Asia',
    'MZ': 'Africa',
    'NA': 'Africa',
    'NC': 'Oceania',
    'NE': 'Africa',
    'NF': 'Oceania',
    'NG': 'Africa',
    'NI': 'North America',
    'NL': 'Europe',
    'NO': 'Europe',
    'NP': 'Asia',
    'NR': 'Oceania',
    'NU': 'Oceania',
    'NZ': 'Oceania',
    'OM': 'Asia',
    'OS': 'Asia',
    'PA': 'North America',
    'PE': 'South America',
    'PF': 'Oceania',
    'PG': 'Oceania',
    'PH': 'Asia',
    'PK': 'Asia',
    'PL': 'Europe',
    'PM': 'North America',
    'PR': 'North America',
    'PS': 'Asia',
    'PT': 'Europe',
    'PW': 'Oceania',
    'PY': 'South America',
    'QA': 'Asia',
    'RE': 'Africa',
    'RO': 'Europe',
    'RS': 'Europe',
    'RU': 'Europe',
    'RW': 'Africa',
    'SA': 'Asia',
    'SB': 'Oceania',
    'SC': 'Africa',
    'SD': 'Africa',
    'SE': 'Europe',
    'SG': 'Asia',
    'SH': 'Africa',
    'SI': 'Europe',
    'SJ': 'Europe',
    'SK': 'Europe',
    'SL': 'Africa',
    'SM': 'Europe',
    'SN': 'Africa',
    'SO': 'Africa',
    'SR': 'South America',
    'SS': 'Africa',
    'ST': 'Africa',
    'SV': 'North America',
    'SY': 'Asia',
    'SZ': 'Africa',
    'TC': 'North America',
    'TD': 'Africa',
    'TG': 'Africa',
    'TH': 'Asia',
    'TJ': 'Asia',
    'TK': 'Oceania',
    'TM': 'Asia',
    'TN': 'Africa',
    'TO': 'Oceania',
    'TP': 'Asia',
    'TR': 'Asia',
    'TT': 'North America',
    'TV': 'Oceania',
    'TW': 'Asia',
    'TZ': 'Africa',
    'UA': 'Europe',
    'UG': 'Africa',
    'US': 'North America',
    'UY': 'South America',
    'UZ': 'Asia',
    'VC': 'North America',
    'VE': 'South America',
    'VG': 'North America',
    'VI': 'North America',
    'VN': 'Asia',
    'VU': 'Oceania',
    'WF': 'Oceania',
    'WS': 'Oceania',
    'XK': 'Europe',
    'YE': 'Asia',
    'YT': 'Africa',
    'ZA': 'Africa',
    'ZM': 'Africa',
    'ZW': 'Africa',
}
```

## Cargamos el dataframe de testeo

```python
hotelsdfTesteoOriginal = pd.read_csv("./hotels_test.csv")
hotelsdfTesteo = hotelsdfTesteoOriginal.copy()
```

# Arbol de decisiones sin optimizacion


Vamos a comenzar creando un arbol de decisiones que tenga en cuenta todas las columnas. \
Luego, vamos a realizar una optimizacion y vamos a optimizar dicho arbol para luego comparar resultados.


## Cargamos nuestro dataframe del checkpoint pasado

Vamos a crear una copia de nuestro dataframe para la creacion del arbol

```python
hotelsdfCheckpoint1 = pd.read_csv("./dataframeCheckpoint1.csv")
hotelsdfArbol = hotelsdfCheckpoint1.copy()
print("El data frame esta compuesto por "f"{hotelsdfArbol.shape[0]}"" filas y "f"{hotelsdfArbol.shape[1]}"" columnas")
```

Un vistazo básico a la información contenida en el dataframe:

```python
pd.concat([hotelsdfArbol.head(2), hotelsdfArbol.sample(5), hotelsdfArbol.tail(2)])
```

Vemos que tenemos una columa extra "Unnamed: 0". Esta hace referencia la columna de origen del registro. Procedemos a borrarla

```python
hotelsdfArbol.drop("Unnamed: 0", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
print()
```

## Transformacion de las columnas para la creacion del arbol


Para poder usar el arbol de sklearn, tenemos que transformar todas nuestras columnas no numericas a valores numericos. \
Dichas columnas son las siguientes:

```python
valoresAConvertir = hotelsdfArbol.dtypes[(hotelsdfArbol.dtypes !='int64') & (hotelsdfArbol.dtypes !='float64')].index
valoresAConvertir = valoresAConvertir.to_list()
valoresAConvertir
```

Sin embargo, no todas estas columnas nos van a servir para nuestro analisis.



### Booking ID


Vamos a empezar removiendo booking\_id visto en como no la necesitamos para analisis

```python
hotelsdfArbol.drop("booking_id", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('booking_id')
```

### Reservation Status & Reservation status date


Reservation Status nos dice el estado de la reservacion, si fue cancelada o no y reservation status date nos marca la fecha en la que cambio el estado. 
Estas dos columnas nos son redundantes

```python
hotelsdfArbol.drop("reservation_status", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('reservation_status')
```

```python
hotelsdfArbol.drop("reservation_status_date", axis=1, inplace=True)
hotelsdfArbol.reset_index(drop=True)
valoresAConvertir.remove('reservation_status_date')
```

### Country
Country toma una amplia cantidad de valores como vimos en el analisis univariado. Asique decidimos agrupar los paises por continente para poder usar la variable

```python
hotelsdfArbol["continente"] = hotelsdfArbol["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfArbol["continente"] = hotelsdfArbol["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfArbol['country'].unique().tolist()
print(country) 
```

```python
country = hotelsdfArbol['continente'].unique().tolist()
print(country) 
```

Viendo estos resultados vemos que hay dos outliers que no logramos identificar en el analisis univariado.


"ATA" refiere al **continente** de Antartida. Al ser un valor tan fuera de lo comun y tener una sola ocurrencia decidimos eliminarlo del dataframe

```python
hotelsdfArbol.drop((hotelsdfArbol[hotelsdfArbol["country"] == "ATA"].index.values),inplace=True)
hotelsdfArbol.reset_index(drop=True)
print() #Este print es para no mostrar el dataframe innecesariamente
```

"UMI" hace referenca a unas islas cerca de Hawaii. Al ser un unico caso y tener una poblacion de 300 habitantes, decidimos considerarlo como Estados Unidos, es decir America del Norte

Fuentes:
- https://www.iso.org/obp/ui/#iso:code:3166:UM
- https://en.wikipedia.org/wiki/United_States_Minor_Outlying_Islands

```python
hotelsdfArbol.loc[hotelsdfArbol['country'] == "UMI", 'country'] = 'North America'
hotelsdfArbol.loc[hotelsdfArbol['continente'] == "UMI", 'continente'] = 'North America'
```

Con estos nuevos cambios, la columna continente toma los siguientes valores

```python
continente = hotelsdfArbol['continente'].unique().tolist()
print(continente) 
```

Procedemos a dropear la columna de country

```python
hotelsdfArbol=hotelsdfArbol.drop(['country'], axis='columns', inplace=False)
valoresAConvertir.remove('country')
valoresAConvertir.append('continente')
hotelsdfArbol.reset_index(drop=True)
```

```python
valoresAConvertir
```

### One hot encoding


Vamos a transformar dichas variables categoricas con la tecnica de one hot encoding. \
Esto va a crear una serie de nuevas columnas con todos los posibles de la variable categorica. En cada columna va a haber un 1 o un 0 para indicar el valor del registro de esa variable. \
Una de las columnas (en este caso la primera) es eliminada ya que, si todas las otras columnas son falsas, significa que la variable toma el valor de la columna eliminada. \
Esto lo podemos hacer gracias a que eliminamos todos nuestros valores faltantes en las secciones anteriores.

```python
#One hot encoding para variables categoricas, esto elimina las columnas categoricas y las reemplaza con el conjunto del hot encoding
hotelsdfArbol = pd.get_dummies(hotelsdfArbol, columns=valoresAConvertir, drop_first=True)
```

Vamos a observar como nos quedo el dataframe despues del one hot encoding

```python
hotelsdfArbol.head()
```

Observamos que hay una **gran** cantidad de columnas


### Aplicamos mismas modificaciones al dataset de testeo


Ahora vamos a aplicar las mismas modificaciones y encodings al dataframe de testeo para poder aplicarle el modelo


Empezamos cambiando el nombre de las columnas para que coincida con el de nuestro dataframe

```python
nuevas_columnas = {
    'adr':'average_daily_rate',
    'adults':'adult_num',
    'agent':'agent_id',
    'arrival_date_day_of_month':'arrival_month_day',
    'arrival_date_month':'arrival_month',
    'arrival_date_week_number':'arrival_week_number',
    'arrival_date_year':'arrival_year',
    'assigned_room_type':'assigned_room_type',
    'babies':'babies_num',
    'booking_changes':'booking_changes_num',
    'children':'children_num',
    'company':'company_id',
    'country':'country',
    'customer_type':'customer_type',
    'days_in_waiting_list':'days_in_waiting_list',
    'deposit_type':'deposit_type',
    'distribution_channel':'distribution_channel',
    'hotel':'hotel_name',
    'id':'booking_id',
    'is_repeated_guest':'is_repeated_guest',
    'lead_time':'lead_time',
    'market_segment':'market_segment_type',
    'meal':'meal_type',
    'previous_bookings_not_canceled':'previous_bookings_not_canceled_num',
    'previous_cancellations':'previous_cancellations_num',
    'required_car_parking_spaces':'required_car_parking_spaces_num',
    'reserved_room_type':'reserved_room_type',
    'stays_in_weekend_nights':'weekend_nights_num',
    'stays_in_week_nights':'week_nights_num',
    'total_of_special_requests':'special_requests_num',
}

hotelsdfTesteo.rename(columns = nuevas_columnas, inplace = True)
```

Antes de nada, vamos a procesar todos los datos faltantes del dataframe.


#### Dias Totales


Anadimos la columna que creamos en el checkpoint 1

```python
hotelsdfTesteo["dias_totales"] = hotelsdfTesteo["week_nights_num"] + hotelsdfTesteo["weekend_nights_num"]
```

#### Datos faltantes

```python
hotelsdfTesteo.isnull().sum()
```

```python
print("Vemos que 'company id' tiene un " + str( (hotelsdfTesteo["company_id"].isnull().sum() * 100) / len(hotelsdfTesteo)  ) + "% de datos faltantes.")
print("Por esto decidimos eliminar la columna (tanto en el dataset de testeo como en el de entrenamiento)")
```

```python
hotelsdfTesteo.drop("company_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
```

### Valores a convertir


Siempre posible, vamos a aplicar el mismo criterio que arriba

```python
valoresAConvertirTesteo = hotelsdfTesteo.dtypes[(hotelsdfTesteo.dtypes !='int64') & (hotelsdfTesteo.dtypes !='float64')].index
valoresAConvertirTesteo = valoresAConvertirTesteo.to_list()
valoresAConvertirTesteo
```

#### Booking ID

```python
hotelsdfTesteo.drop("booking_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('booking_id')
```

#### Agent ID


Tomamos el mismo criterio que el checkpoint 1. Transformamos a 0

```python
hotelsdfTesteo.loc[hotelsdfTesteo['agent_id'].isnull(), 'agent_id'] = 0
```

#### Reservation Status & Reservation status date



Dropeamos estas columnas debido a que no nos dan ninguna informacion adicional

```python
hotelsdfTesteo.drop("reservation_status_date", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('reservation_status_date')
```

#### Country y Continents


Para los valores faltantes de country tomamos el mismo criterio del checkpoint 1. Los convertimos en portugal

```python
hotelsdfTesteo.loc[hotelsdfTesteo['country'].isnull(), 'country'] = 'PRT'
```

```python
hotelsdfTesteo["continente"] = hotelsdfTesteo["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfTesteo["continente"] = hotelsdfTesteo["continente"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfTesteo['country'].unique().tolist()
valoresAConvertirTesteo.append("continente")
print(country) 
```

```python
continentes = hotelsdfTesteo['continente'].unique().tolist()
print(continentes) 
```

Tal como ocurrio con el dataset de Train, observamos que hay algunos continente (y por tanto sus paises y registros asociados) que parecen ser outliers.
Los estudiamos

```python
hotelsdfTesteo[ hotelsdfTesteo['continente'] =="ATA"]
```

Hay un registro correspondiente a "antartida". como no podemos dropearlo, le ponemos de continente "north america".\
Le asignamos el valor de America del norte debido a que estados unidos es el pais con mas bases en la antartica

**TODO:CHEUQUEAR LO DE ARRIBA**

```python
hotelsdfTesteo.loc[hotelsdfTesteo['continente'] == "ATA", 'continente'] = "North America"
```

```python
hotelsdfTesteo[ hotelsdfTesteo['continente'] =="ATF"]
```

"ATF", que es la sigla de Fr. So. Ant. Tr (French southern and antartic lands).
Ponemos su contienente en Europa. 

```python
hotelsdfTesteo.loc[hotelsdfTesteo['continente'] == "ATF", 'continente'] = "Europe"
```

```python
hotelsdfTesteo[hotelsdfTesteo['continente'] =="ATF"]
```

#### Analisis de valores faltantes de continente

```python
hotelsdfTesteo[hotelsdfTesteo['continente'].isna()]
```

Vemos que hay una serie de registros que no tienen el dato del pais. Sin embargo, no son muchos. Debido a esto, vamos a asignarle estos registros el valor de aquel contiente que tenga la mayor cantidad de registros

```python
sns.countplot(data = hotelsdfTesteo, x = 'continente', palette= 'Set2')
plt.title('Cantidad de registros por continente')
plt.xlabel('Continente')
plt.ylabel('Cantidad de registros')
```

Vemos que el continente con mayor cantidad de registros es europa, asique lo asignamos a ese valor

```python
#hotelsdfTesteo.loc[hotelsdfTesteo['continente'].isna()] = "Europe"
hotelsdfTesteo.loc[hotelsdfTesteo['continente'].isnull(), 'country'] = 'Europe'
```

Miro q se hayan cambiado bien todos los continentes y no haya valores raros

```python
continentes = hotelsdfTesteo['continente'].unique().tolist()
print(continentes)
#OJO CON EL NAN
```

Como hicimos con el dataset de train, y ya habiendo procesado la columna continente, dropeamos la columna country

```python
hotelsdfTesteo=hotelsdfTesteo.drop(['country'], axis='columns', inplace=False)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertirTesteo.remove('country')
```

#### previous bookings not cancelled


Al igual q en el train, dropeamos esta col

```python
hotelsdfTesteo=hotelsdfTesteo.drop(['previous_bookings_not_canceled_num'], axis='columns', inplace=False)
hotelsdfTesteo.reset_index(drop=True)
```

```python
hotelsdfTesteo.isnull().sum() #AMONGUS
```

### One hot encoding del testeo

```python
#One hot encoding para variables categoricas, esto elimina las columnas categoricas y las reemplaza con el conjunto del hot encoding
hotelsdfTesteo = pd.get_dummies(hotelsdfTesteo, columns=valoresAConvertirTesteo, drop_first=True)
hotelsdfTesteo.head()
```

### Corroboracion de columnas


Despues de todas estas transformaciones vamos a corrobar que los dataframes tengan la misma cantidad de columnas.

```python
set_test = set(hotelsdfTesteo.columns)
set_arbol = set(hotelsdfArbol.columns)

missing = list(sorted(set_test - set_arbol))
added = list(sorted(set_arbol - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

Vemos que en el dataframe del arbol nos sobra la columna "is canceled", cosa que hace sentido ya que esa es la columna con la que vamos a entrenar al dataset. Sin embargo, vemos que tambien hay 3 columnas que faltan en el dataset de arbol. 


Vamos a reasignar los valores de las columnas de test para que coincidan.

El siguiente codigo nos calcula cuantas personas tiene cada tipo de cuarto

```python
cantDeCuartos = {}
cantidadDeCasosSumados = 0

cantDeCuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipoDeCuarto = 'reserved_room_type_' + letra
    cantidadDeCasosSumados += 1
    if tipoDeCuarto not in hotelsdfTesteo.columns:
        continue
    hotelsdfTesteo[tipoDeCuarto]
    #print("SSUS")
    resultado = hotelsdfTesteo[hotelsdfTesteo[tipoDeCuarto] == 1][tipoDeCuarto].sum()
    cantDeCuartos[letra] = resultado

cuartosA = len(hotelsdfTesteo) - cantidadDeCasosSumados
cantDeCuartos["A"] = cuartosA


cantDeCuartos
```

Vemos que L y P tienen una extremadamente pequena cantidad de apariciones. \
Lo vamos a anadir al roomtype A al ser el que tiene la mayor cantidad de apariciones.

Para anadirlos a la columna a, simplemente tenemos que eliminar las columnas L y P (ya que la columna A es la eliminada por el one hot)

```python
hotelsdfTesteo.drop("reserved_room_type_L", axis=1, inplace=True)
hotelsdfTesteo.drop("reserved_room_type_P", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
print()
```

Vamos a aplicar el mismo criterio a assigned room type

```python
cantDeCuartos = {}
cantidadDeCasosSumados = 0

cantDeCuartos["A"] = 0 #Arrancamos asignado 0 a los cuartos de A. Estos fueron removidos por el one hot. Lo vamos a calcular al final.
for letra in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']:
    tipoDeCuarto = 'assigned_room_type_' + letra
    cantidadDeCasosSumados += 1
    if tipoDeCuarto not in hotelsdfTesteo.columns:
        continue
    hotelsdfTesteo[tipoDeCuarto]
    #print("SSUS")
    resultado = hotelsdfTesteo[hotelsdfTesteo[tipoDeCuarto] == 1][tipoDeCuarto].sum()
    cantDeCuartos[letra] = resultado

cuartosA = len(hotelsdfTesteo) - cantidadDeCasosSumados
cantDeCuartos["A"] = cuartosA


cantDeCuartos
```

Aca tambien vemos que P tiene muy pocas aparciones. Asique aplicamos el mismo criterio de antes

```python
hotelsdfTesteo.drop("assigned_room_type_P", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
print()
```

Vemos ahora que nuestras columnas coinciden

```python
set_test = set(hotelsdfTesteo.columns)
set_arbol = set(hotelsdfArbol.columns)

missing = list(sorted(set_test - set_arbol))
added = list(sorted(set_arbol - set_test))

print('Faltan en arbol:', missing)
print('Sobran en arbol:', added)
```

## Entrenamiento del modelo

```python
#Creamos un dataset con los features que vamos a usar para tratar de predecir el target
hotelsdfArbol_x=hotelsdfArbol.drop(['is_canceled'], axis='columns', inplace=False)

#Creo un dataset con la variable target
hotelsdfArbol_y = hotelsdfArbol['is_canceled'].copy()

#Genero los conjuntos de train y de test
x_train, x_test, y_train, y_test = train_test_split(hotelsdfArbol_x,
                                                    hotelsdfArbol_y, 
                                                    test_size=0.2,  #proporcion 80/20
                                                    random_state=9) #usamos la semilla 9 porque somos el grupo 9
```

Ahora ya tenemos generados nuestros conjuntos de train y test; y tenemos nuesto dataframe con los datos numericos vamos a generar nuestro modelo

```python
#Vamos a iniciar con una profundidad maxima considerable para tener un arbol de dimesiones considerables
PROFUNDIDAD_MAX = 20

#Creamos un clasificador con hiperparámetros 
tree_model = tree.DecisionTreeClassifier(criterion="gini", #Gini es el criterio por defecto
                                         max_depth = PROFUNDIDAD_MAX) 

#Entrenamos el modelo con el conjunto de entrenamiento
model = tree_model.fit(X = x_train, y = y_train)
```

```python
#Realizamos una predicción sobre el set de test
y_pred = model.predict(x_test)
#Valores Predichos
y_pred
```

```python
ds_resultados=pd.DataFrame(zip(y_test,y_pred),columns=['test','pred'])
ds_resultados
```

Estas columns representan 20% de nuestro dataframe que fue dedicado al testeo del modelo


Vamos a graficar la matriz de confusion para visualizar los resultados de nuesto modelo:

```python
#Creo la matriz de confusión
tabla=confusion_matrix(y_test, y_pred)

#Grafico la matriz de confusión
sns.heatmap(tabla,cmap='GnBu',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

A continuacion vamos a graficar el arbol resultante: \
(Advertencia: Suele tardar unos minutos en terminar de renderizar la imagen)

```python
plt.figure(figsize=(100,100))

tree_plot_completo=tree.plot_tree(model,
                                 feature_names=hotelsdfArbol_x.columns.to_list(),
                                 filled=True,
                                 rounded=True,
                                 class_names=['Not Canceled','Is canceled']) #model.classes_
plt.show(tree_plot_completo)
```

Con la imagen se ve que el arbol resultante tiene unas dimensiones exageradas, vemos ademas que tiene una profundidad de 20 como especificamos


Vemos que, sin ningun tipo de optimizacion y con un arbol de profundidad 20 y sin ningun tipo de poda obtenemos, en nuestro dataset de testeo:

```python
accuracy=accuracy_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred,)
precision=precision_score(y_test,y_pred)

print("Accuracy: "+str(accuracy))
print("Recall: "+str(recall))
print("Precision: "+str(precision))
print("f1 score: "+str(f1))
```

## Randomized Serach Cross Validation


### OJOOOOO TODOOO PUSE 2, 50 NO ES UNA BANDA!!!??

```python
##KFOLD CV Random Search para buscar el mejor arbol (los mejores atributos, hiperparametros,etc)

#Cantidad de combinaciones que quiero porbar
n=10

#Conjunto de parámetros que quiero usar
params_grid = {'criterion':['gini','entropy'],
               #'min_samples_leaf':list(range(2,12)), #cantidad de datos que puede tener una hoja
               #'min_samples_split': list(range(2,20)), #cantidad de datos que puede tener un nodo
               'ccp_alpha':np.linspace(0,0.0007,n), #poda
               #(0,0.0007)
               'max_depth':list(range(2,50))}
                #CON 2, 50 ---> F1 SCORE = 0,81

#-------OJOOOOO TODO PUSE 2, 50 NO ES UNA BANDA!!!??                
#CON 10 ---> F1 SCORE = 0,75 q es mas bajo q el arbol gigante....

#Cantidad de splits para el Cross Validation
folds=5

#Kfold estratificado
kfoldcv = StratifiedKFold(n_splits=folds)

#Clasificador
base_tree = DecisionTreeClassifier() 

#Metrica que quiero optimizar F1 Score
scorer_fn = make_scorer(sk.metrics.f1_score)

#Random Search Cross Validation
randomcv = RandomizedSearchCV(estimator=base_tree,
                              param_distributions = params_grid,
                              scoring=scorer_fn,
                              cv=kfoldcv,
                              n_iter=n) 

#Busco los hiperparamtros que optimizan F1 Score
randomcv.fit(x_train,y_train)
```

Mostramos los mejores hiperparametros devueltos por el arbol y el valor del f1_score

```python
#Mejores hiperparametros del arbol
print(randomcv.best_params_)
#Mejor métrica
print("f1_score = ",randomcv.best_score_)
```

Algunos valores obtenidos del algorimo

```python
randomcv.cv_results_['mean_test_score']
```

Atributos considerados y su importancia

```python
#Atributos considerados y su importancia
#TODO poner features considerados con una minu explicacion antes de fabricacion de arbol
features_considerados = hotelsdfArbol_x.columns.to_list()

best_tree = randomcv.best_estimator_
feat_imps = best_tree.feature_importances_

for feat_imp,feat in sorted(zip(feat_imps,features_considerados)):
  if feat_imp>0:
    print('{}: {}'.format(feat,feat_imp))
```

## Predicción y Evaluación del Modelo con mejores hiperparámetros


Creo el árbol con los mejores hiperparámetros

```python
#Creo el árbol con los mejores hiperparámetros
#TODO "Mostrar una porcion significativa"

from sklearn.tree import export_text

arbol_mejores_parametros=DecisionTreeClassifier().set_params(**randomcv.best_params_)

#Entreno el arbol en todo el set
arbol_mejores_parametros.fit(x_train,y_train)

reglas = export_text(arbol_mejores_parametros, feature_names=list(features_considerados))
print(reglas)
```

### Arbol pero mas lindo
###TODO SI se desea hacer funcar la libreria


```python
# from six import StringIO
# from IPython.display import Image  
# from sklearn.tree import export_graphviz
# import pydotplus
# import matplotlib.pyplot as plt

# dot_data = StringIO()
# export_graphviz(arbol, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True,
#                 feature_names=features_considerados,
#                 class_names=['good','bad'])

# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())
```

### Prediccion con split de train


Evalúo el Arbol con los mejores hiperparámetros

```python
#Evalúo el Arbol con los mejores hiperparámetros

#Hago predicción sobre el set de evaluacion
y_pred= arbol_mejores_parametros.predict(x_test)

#Arbol Reporte y Matriz de Confusion
#print(classification_report(y_test,y_pred))
print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) #binary considera la clase positiva por defecto 1

cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')

#TODO Mostrar todas las metricas xa q quede mas lindo
```

Muestro array de predcciones

```python
arbol_mejores_parametros.predict_proba(x_test)
```

## Entrenamiento Cross Validation

```python

#Spits que respeten la proporción delas clases
#TODO
kfoldcv =StratifiedKFold(n_splits=10) 

#Selecciono métrica F1-Score (misma que antes)
scorer_fn = make_scorer(sk.metrics.f1_score)

#Hago CV
#resultados = cross_validate(arbolcv,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

resultados = cross_validate(arbol_mejores_parametros,x_train, y_train, cv=kfoldcv,scoring=scorer_fn,return_estimator=True)

metricsCV = resultados['test_score']

arbol_mejor_performance = resultados['estimator'][np.where(metricsCV==max(metricsCV))[0][0]]

```

```python
#Grafico Boxplot -Entrenado con 50 Fold Cross Validation

metric_labelsCV = ['F1 Score']*len(metricsCV) 
sns.set_context('talk')
sns.set_style("darkgrid")
plt.figure(figsize=(8,8))
sns.boxplot(metricsCV)
#sns.boxplot(metric_labelsCV,metricsCV)
```

### Reglas del arbol


### Grafico del arbol


## Predicción y Evaluación del Modelo

```python
#Arbol CV set de evaluación

#Predicción sobre el set de evaluacion
y_pred= arbol_mejor_performance.predict(x_test)


#Arbol Reporte y Matriz de Confusion
print(classification_report(y_test,y_pred))

print('F1-Score: {}'.format(f1_score(y_test, y_pred, average='binary'))) #binary considera la clase positiva por defecto 1


cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm, cmap='Blues',annot=True,fmt='g')
plt.xlabel('Predicted')
plt.ylabel('True')
```

## Ahora vamos a compararlo con el dataset de testeo de verdad

```python
#Realizamos una predicción sobre el set de test
#y_pred = model.predict(hotelsdfTesteo)
#la de abajo #0,81
y_pred= arbol_mejor_performance.predict(hotelsdfTesteo)

#Valores Predichos
len(y_pred)
```

```python
df_submission = pd.DataFrame({'id': hotelsdfTesteoOriginal['id'], 'is_canceled': y_pred})
df_submission.head()
```

```python
df_submission.to_csv('submissions/arbol_decisiones_ineficiente.csv', index=False)
```

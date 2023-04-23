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
hotelsdfTesteo = pd.read_csv("./hotels_test.csv")

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
Country toma una amplia cantidad de valores como vimos en el analisis univariado. Asique decidimos agrupar los paises por continentes para poder usar la variable

```python
hotelsdfArbol["Continentes"] = hotelsdfArbol["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfArbol["Continentes"] = hotelsdfArbol["Continentes"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfArbol['country'].unique().tolist()
print(country) 
```

```python
country = hotelsdfArbol['Continentes'].unique().tolist()
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
hotelsdfArbol.loc[hotelsdfArbol['Continentes'] == "UMI", 'Continentes'] = 'North America'
```

Con estos nuevos cambios, la columna Continentes toma los siguientes valores

```python
continentes = hotelsdfArbol['Continentes'].unique().tolist()
print(continentes) 
```

Procedemos a dropear la columna de country

```python
hotelsdfArbol=hotelsdfArbol.drop(['country'], axis='columns', inplace=False)
valoresAConvertir.remove('country')
valoresAConvertir.append('Continentes')
hotelsdfArbol.reset_index(drop=True)
```

### One hot encoding


Vamos a transformar dichas variables categoricas con la tecnica de one hot encoding. \
Esto va a crear una serie de nuevas columnas con todos los posibles de la variable categorica. En cada columna va a haber un 1 o un 0 para indicar el valor del registro de esa variable. \
Una de las columnas (en este caso la primera) es eliminada ya que, si todas las otras columnas son falsas, significa que la variable toma el valor de la columna eliminada. \
Esto lo podemos hacer gracias a que eliminamos todos nuestros valores faltantes en las secciones anteriores.

```python
#One hot encoding para variables categoricas, esto elimina las columnas categoricas y las reemplaza con el conjunto del hot encoding
hotelsdfArbol = pd.get_dummies(hotelsdfArbol, columns=valoresAConvertir, drop_first=True)
hotelsdfArbol.head()
```

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


#### Datos faltantes

```python
hotelsdfTesteo.isnull().sum()
```

##### company_id

```python
print("Vemos que 'company id' tiene un " + str( (hotelsdfTesteo["company_id"].isnull().sum() * 100) / len(hotelsdfTesteo)  ) + "% de datos faltantes.")
print("Por esto decidimos eliminar la columna (tanto en el dataset de testeo como en el de entrenamiento)")
```

```python
hotelsdfTesteo.drop("company_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)

#hotelsdfArbol.drop("company_id", axis=1, inplace=True)
#hotelsdfArbol.reset_index(drop=True)
#Nosotros ya teniamos company_id dropeado del checkpoint anterior
```

##### agent id


Vamos a aplicar el mismo criterio que en el checkpoint 1

```python
#Reemplazamos valores faltantes por 0 ya que no existe previamente y servira para regular los tipos de atos de la columna
hotelsdfTesteo.loc[hotelsdfTesteo['agent_id'].isnull(), 'agent_id'] = 0
hotelsdfTesteo[hotelsdfTesteo.agent_id.isnull()]
hotelsdfTesteo['agent_id'] = hotelsdfTesteo['agent_id'].astype(int)
```

##### country


Vamos a aplicar el mismo criterio que en el checkpoint 1

```python
#Para evitar la eliminacion de los registros y debido a la muy marcada tendencia de las reservas a venir de Portugal.
hotelsdfTesteo.loc[hotelsdfTesteo['country'].isnull(), 'country'] = 'PRT'
```

#### Transformamos las columnas cualitativas a numericas


Tomamos el mismo criterio que en la seccion sobre el set de entrenamiento

```python
valoresAConvertir = hotelsdfTesteo.dtypes[(hotelsdfTesteo.dtypes !='int64') & (hotelsdfTesteo.dtypes !='float64')].index
valoresAConvertir = valoresAConvertir.to_list()
valoresAConvertir
```

```python
hotelsdfTesteo.drop("booking_id", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertir.remove('booking_id')
```

```python
hotelsdfTesteo.drop("reservation_status_date", axis=1, inplace=True)
hotelsdfTesteo.reset_index(drop=True)
valoresAConvertir.remove('reservation_status_date')
```

```python
hotelsdfTesteo["Continentes"] = hotelsdfTesteo["country"].replace(COUNTRY_ALPHA3_TO_COUNTRY_ALPHA2)
hotelsdfTesteo["Continentes"] = hotelsdfTesteo["Continentes"].replace(COUNTRY_ALPHA2_TO_CONTINENT)
```

```python
country = hotelsdfTesteo['country'].unique().tolist()
print(country) 
```

```python
country = hotelsdfTesteo['Continentes'].unique().tolist()
print(country) 
```

"ATF" hace referencia a islas francesas
"TMP" hace referencia a timor oriental
- https://www.iso.org/obp/ui#iso:code:3166:FQHH
- https://www.iso.org/obp/ui#iso:code:3166:TP

```python

hotelsdfArbol.loc[hotelsdfArbol['country'] == "UMI", 'country'] = 'North America'
hotelsdfArbol.loc[hotelsdfArbol['Continentes'] == "UMI", 'Continentes'] = 'North America'
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

Ahora vamos a compararlo con el dataset de testeo de verdad

```python
#Creamos un dataset con los features que vamos a usar para tratar de predecir el target
#hotelsdfArbol_x=hotelsdfArbol.drop(['is_canceled'], axis='columns', inplace=False)

#Creo un dataset con la variable target
#hotelsdfTesteo_predecir = hotelsdfTesteo['is_canceled'].copy()

#Genero los conjuntos de train y de test
#x_train, x_test, y_train, y_test = train_test_split(hotelsdfArbol_x,
#                                                    hotelsdfArbol_y, 
#                                                    test_size=0.2,  #proporcion 80/20
#                                                    random_state=9) #usamos la semilla 9 porque somos el grupo 9
```

```python
#Realizamos una predicción sobre el set de test
#y_pred = model.predict(hotelsdfTesteo)
#Valores Predichos
#y_pred
```

```python
hotelsdfTesteo
```

```python
hotelsdfArbol
```

```python

```

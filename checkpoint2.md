# Arbol de decisiones


Vamos a comenzar creando un arbol de decisiones que tenga en cuenta todas las columnas. \
Luego, vamos a realizar una poda y vamos a optimizar dicho arbol para luego comparar resultados.


# Importamos


SUSUSUSUSU


AMONGUS


## Creamos un nuevo Data frame


Para esto vamos a crear una copia de nuestro dataframe para la creacion del arbol

```python
hotelsdfArbol = hotelsdf.copy()
```

## Transformacion de las columnas para la creacion del arbol


Para poder usar el arbol de sklearn, tenemos que transformar todas nuestras columnas no numericas a valores numericos. \
Dichas columnas son las siguientes:


AMONGUS

```python
valoresAConvertir = hotelsdf.dtypes[(hotelsdf.dtypes !='int64') & (hotelsdf.dtypes !='float64')].index
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
country = hotelsdf['country'].unique().tolist()
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
country = hotelsdfArbol['Continentes'].unique().tolist()
print(country) 
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

```python

```

```python
hotelsdfArbol.head()
```

```python
hotelsdfArbol
```

```python
#Creo un clasificador
tree_model = tree.DecisionTreeClassifier(max_depth = 10)

#Entreno el modelo
model = tree_model.fit(X = x_train, y = y_train) 
```

```python
model
```

IMPOSTOR AMONGUS


BIG CHUNGUS

```python

```

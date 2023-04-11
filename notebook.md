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

En este jupyter notebook vamos a explorar un conjunto de datos sobre reservas de hoteles y tratar de hallar un modelo que nos permita predecir si la reserva va a ser cancelada

# Preparacion del ambiente de trabajo
Importamos todas las librerias que vamos a usar


```python
import pandas as pd 
import numpy as np
import sklearn as sk
import seaborn as sns
from matplotlib import pyplot as plt
import scipy.stats as st
from calendar import month_name
```

## Cargamos de datos a un dataframe

Se carga la información a un dataframe de pandas, se genera una copia y se trabaja sobre la misma

```python
hotelsDfOriginal = pd.read_csv("./hotels_train.csv")
hotelsdf = hotelsDfOriginal.copy()

print("El data frame esta compuesto por "f"{hotelsdf.shape[0]}"" filas y "f"{hotelsdf.shape[1]}"" columnas")
```

## Vistazo inicial

Un vistazo básico a la información contenida en el dataframe:

```python
pd.concat([hotelsdf.head(2), hotelsdf.sample(5), hotelsdf.tail(2)])
```


Renombramos las columnas del dataframe con nombres mas claros

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
    'is_canceled':'is_canceled',
    'is_repeated_guest':'is_repeated_guest',
    'lead_time':'lead_time',
    'market_segment':'market_segment_type',
    'meal':'meal_type',
    'previous_bookings_not_canceled':'previous_bookings_not_canceled_num',
    'previous_cancellations':'previous_cancellations_num',
    'required_car_parking_spaces':'required_car_parking_spaces_num',
    'reservation_status':'reservation_status',
    'reservation_status_date':'reservation_status_date',
    'reserved_room_type':'reserved_room_type',
    'stays_in_weekend_nights':'weekend_nights_num',
    'stays_in_week_nights':'week_nights_num',
    'total_of_special_requests':'special_requests_num',
}

hotelsdf.rename(columns = nuevas_columnas, inplace = True)
```

Por otro lado, podemos observar que tipo de dato almacena cada columna y cuales tienen datos faltantes

```python
hotelsdf.info()
```

# Analisis de variables
Vamos a dividir las variables en cuantitativas y cualitativas.

|     Nombre de la variable           |       Tipo      |      Descripcion         |
| ----------------------------------- | --------------- | ------------------------ |  
| average_daily_rate                  | Cuantitativa    | Promedio de la ganancia diaria, por habitacion                              |
| adult_num                           | Cuantitativa    |           numero de adultos en la reserva              |
| agent_id                            | Cualitativa     |                          |
| arrival_month_day                   | Cuantitativa    |                          |
| arrival_month                       | Cualitativa     |                          |
| arrival_week_number                 | Cuantitativa    |                          |
| arrival_year                        | Cuantitativa    |                          |
| assigned_room_type                  | Cualitativa     |                          |
| babies_num                          | Cuantitativa    |                          |
| booking_changes_num                 | Cuantitativa    |                          |
| children_num                        | Cuantitativa    |                          |
| company_id                          | Cualitativa     |                          |
| country                             | Cualitativa     |                          |
| customer_type                       | Cualitativa     |                          |
| days_in_waiting_list                | Cuantitativa    |                          |
| deposit_type                        | Cualitativa     |                          |
| distribution_channel                | Cualitativa     |                          |
| hotel_name                          | Cualitativa     |                          |
| is_repeated_guest                   | Cualitativa     |                          |
| lead_time                           | Cuantitativa    |                          |
| market_segment_type                 | Cualitativa     |                          |
| meal_type                           | Cualitativa     |                          |
| previous_bookings_not_canceled_num  | Cuantitativa    |                          |
| previous_cancellations_num          | Cuantitativa    |                          |
| required_car_parking_spaces_num     | Cuantitativa    |                          |
| reservation_status                  | Cualitativa     |                          |
| reservation_status_date             | Cuantitativa    |                          |
| reserved_room_type                  | Cualitativa     |                          |
| weekend_nights_num                  | Cuantitativa    |                          |
| week_nights_num                     | Cuantitativa    |                          |
| special_requests_num                | Cuantitativa    |                          |


## Cuantitativas

Se trabaja inicialmente sobre las variables que han sido identificadas como cuantitativas, se grafican y se intenta realizar la identificación de outliers, por otro lado, aquellas que de un analisis exploratorio previo arrojaron la existencia de *nulls/nans* se realiza algún tipo de reemplazo por el valor más conveniente

Creamos una lista con todas las variables cuantitativas

```python
cuantitativas = [
"adult_num",
"arrival_month_day",
"arrival_week_number",
"arrival_year",
"average_daily_rate",
"babies_num",
"booking_changes_num",
"children_num",
"days_in_waiting_list",
"lead_time",
"previous_bookings_not_canceled_num",
"previous_cancellations_num",
"required_car_parking_spaces_num",
"reservation_status_date",
"special_requests_num",
"weekend_nights_num",
"week_nights_num",
]
```
### Adult number 

Realizamos un analisis sobre la variable adult number

#### Valores estadisticos relevantes 

```python
hotelsdf.adult_num.describe()
```

Dentro de los parametros estadisticos representativos observamos un minimo de 0 adultos y un maximo de 55, ambos representando registros con valores anormales. 

#### Grafica de distribucion

Para mas informacion sobre la frecuencia de los valores se grafican las frecuencias

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

#### Outliers

```python
a_eliminar_con_cero = hotelsdf[hotelsdf['adult_num'] == 0]
a_eliminar_con_cero 
```

```python
a_eliminar_mayores_3 = hotelsdf[hotelsdf['adult_num'] > 3]
a_eliminar_mayores_3
```

```python
print(f'Total de registros a eliminar: {len(a_eliminar_con_cero) + len(a_eliminar_mayores_3)}')
```

Existen 41 registros con valores superiores a 3, los cuales representan outliers. A su vez, se incluyen a estos registros aquellos valores identificados previamente 

#### Ajustes de valor

Eliminamos dichos valores que representan un porcentaje infimo y pueden llegar a desviar las futuras predicciones

```python
hotelsdf.drop(a_eliminar_con_cero.index, inplace = True)
hotelsdf.drop(a_eliminar_mayores_3.index, inplace = True)
hotelsdf.reset_index()
hotelsdf[(hotelsdf["adult_num"] > 4) | hotelsdf['adult_num'] == 0]
```

Por otro lado realizamos de nuevo las graficas de la distribucion para verificar que no cambie significativamente

```python
sns.countplot(data = hotelsdf, x = 'adult_num', palette= 'Set2')
plt.title('Cantidad de adultos por reserva')
plt.xlabel('Numero de adultos')
plt.ylabel('Frecuencia')
```

### arrival month day

#### Valores estadisticos relevantes

```python
hotelsdf["arrival_month_day"].describe()
```

Los parametros estadisticos relevantes no muestran por si solos valores irregulares en el analisis

#### Grafica de distribucion

Una grafica puede llegar a esclarecer o identificar valores fuera de lo comun dentro del dataframe

```python
plt.figure(figsize=(8,4))
sns.countplot(data = hotelsdf, x = 'arrival_month_day')
plt.title("Dia de llegada del mes")
plt.xlabel(xlabel = 'Dia de llegada')
plt.ylabel(ylabel = 'Frecuencia')
```

El analisis univariado de arrival month day no arroja informacion relevante, pero por otro lado, muestra que la variable no presenta ningun valor inesperado y desmuestra que no hay un dia de predilecto del mes

```python
plt.xlabel(xlabel = 'Dia de llegada')
sns.boxplot(data = hotelsdf['arrival_month_day'])
plt.title("Dia de llegada del mes")
plt.ylabel(ylabel = 'Frecuencia')
```
Por lado un boxplot afirma las concluciones derivadas del grafico anterior 


### arrival week number 

#### Valores estadisticos relevantes

```python
hotelsdf.arrival_week_number.describe()
```
Un vistazo inicial a los parametros estadisticos no muestra inconsistencias en los registros

#### Grafica de distribucion

```python
plt.figure(figsize=(15,5))
sns.countplot(data = hotelsdf, x = 'arrival_week_number', palette='Set2')
plt.title('Semanas del año')
plt.xlabel('Numero de la semana')
plt.ylabel('Frecuencia')
```
De la grafica concluimos que no existen outliers entre los registros 

### arrival year 

#### Valores estadisticos relevantes

```python
hotelsdf.arrival_year.describe()
```

#### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x = 'arrival_year')
plt.xlabel('Años')
plt.ylabel('Frecuencia')
plt.title('Año de las reservas')
```
Todos los registros corresponden a los años: 2015, 2016 y 2017 siendo el año 2016 el mas frecuente entre los registros

### Average Daily Rate

Realizamos un analisis sobre la variable average daily rate

#### Valores estadisticos relevantes 

```python
hotelsdf.average_daily_rate.describe()
```

#### Grafica de distribucion

```python
sns.kdeplot(data = hotelsdf.average_daily_rate)
plt.xlabel(xlabel = 'Montos')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del Precio promedio de renta diaria')
```

#### Outliers

Del grafico anterior se observan registros de average daily rate los cuales tienen asignados 0, se debe estudiar a que se deben esos valores, asi como tambien tratar el valor negativo que aparece como mínimo, por otro lado, analizamos cuantos de los precios presentes en los registros presentan una desviacion considerable de los valores esperados

```python
sns.boxplot(data = hotelsdf['average_daily_rate'])
plt.title("Precio promedio de renta diaria")
plt.xlabel('Average daily rate')
plt.ylabel('Montos')
```

```python

valores_con_cero = len(hotelsdf[hotelsdf['average_daily_rate'] <= 0])
total_valores = len(hotelsdf.average_daily_rate)
porcentaje_con_cero = valores_con_cero/total_valores
print(f" Los de adrs que registran un valor de 0 representa un porcentaje de:{porcentaje_con_cero}' por lo tanto considerando que no son representativos, eliminamos dichos registros inconsistentes ")
```

eliminar valores con 0

```python
a_eliminar_con_cero = hotelsdf[hotelsdf['average_daily_rate'] <= 0].index
hotelsdf.drop(a_eliminar_con_cero, inplace = True)
```

#### Ajustes de valor


Utilizamos Z-score para clasificar las desviasiones presentes en los valores


```python

media_requisitos=np.mean(hotelsdf.average_daily_rate)

std_ard=np.std(hotelsdf.average_daily_rate)

hotelsdf['z_adr']=(hotelsdf.average_daily_rate - media_requisitos)/std_ard

hotelsdf['z_adr']=st.zscore(hotelsdf.average_daily_rate)
hotelsdf[(hotelsdf['z_adr'] > 3) | (hotelsdf['z_adr'] < -2)]
```

Graficamos el Z-score del adr


```python
plt.hist(hotelsdf.z_adr)
plt.title('Histograma Z-Score req')
plt.xlabel('Z-Score req')
plt.xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])
plt.show()
```

```python
desviacion_uno = hotelsdf[(hotelsdf['z_adr'] > 3)]
desviacion_dos = hotelsdf[(hotelsdf['z_adr'] < -2)]
hotelsdf.drop(desviacion_uno.index, inplace = True)
hotelsdf.drop(desviacion_dos.index, inplace = True)
hotelsdf.reset_index()
```

```python
total_valores = len(hotelsdf.average_daily_rate)
cantidad_a_eliminar = desviacion_uno.average_daily_rate.count() + desviacion_dos.average_daily_rate.count()
print("Vamos a eliminar " + str(cantidad_a_eliminar)  + " valores ya son valores que tienen una desviacion estandar muy marcada con  respecto al resto de los valores. Ademas, estos valores representan un " +  str(cantidad_a_eliminar/total_valores) + " porcentaje del total")
```

Graficamos nuevamente con el proposito de verificar la nueva distribucion adquirida luego de la modificacion 



```python
data = hotelsdf.average_daily_rate
sns.kdeplot(data = data)
plt.xlabel(xlabel = 'Average daily rate')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del average daily rate')
```

```python
hotelsdf.drop(labels = 'z_adr', inplace = True, axis = 1)
```

### babies number 

#### Valores estadisticos relevantes


```python
hotelsdf.babies_num.describe()
```

#### Grafica de distribucion

```python
eje_y = hotelsdf.babies_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de bebes')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de bebes por reserva')
```

#### Outliers

```python
hotelsdf[(hotelsdf.babies_num >= 1) & (hotelsdf.adult_num < 1)]
```

#### Ajustes de valor

```python
hotelsdf.drop(hotelsdf[hotelsdf.babies_num == 9].index, inplace = True)
hotelsdf.reset_index()
```

### booking changes number 

#### Valores estadisticos relevantes

```python
hotelsdf.booking_changes_num.describe()
```

#### Grafica de distribucion

```python
eje_y = hotelsdf.booking_changes_num.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Numero de cambios')
plt.ylabel(ylabel='Frecuencia')
plt.title('Cantidad de cambios por reserva')
```
#### Outliers

```python

```

#### Ajustes de valor

```python

```

### children number 

#### Valores estadisticos relevantes
#### Grafica de distribucion

```python
hotelsdf["children_num"].describe()
```
Children number representa la cantidad de niños que fueron registrados en la reserva.\
Esta variable es **discreta**, porque representa una cantidad discreta de niños.\
Sin embargo, esta almacenada como float64 porque tiene valores faltantes.


#### Valores nulos/faltantes

```python
hotelsdf.children_num.isna().sum()
```

Vemos que tenemos 4 valores faltantes.
Vamos a ver cuales son

```python
hotelsdf[hotelsdf["children_num"].isna() == True]
```

```python
cantidadFilas = len(hotelsdf.index)
cantidadDeChildrenNumVacios = hotelsdf.children_num.isna().sum()
print("Considerando que la cantidad de datos de children_num faltante es " + str((cantidadDeChildrenNumVacios * 100) / cantidadFilas) + "%, lo podemos borrar")
```

```python
# Borramos las columnas sin valores
hotelsdf.drop((hotelsdf[hotelsdf["children_num"].isna() == True].index.values),inplace=True)
```

```python
# Casteamos la columna de children number a int, ahora que ya no tiene los valores nana
hotelsdf = hotelsdf.astype({'children_num':'int'})
```

```python
# Corroboramos que el casteo funciono
print(hotelsdf["children_num"].dtypes)
```

#### Outliers

```python
eje_y = hotelsdf["children_num"].value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de ninos')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de ninos por reserva')

hotelsdf["children_num"].value_counts()
```

Vemos que la gran mayoria de las reservas fueron hechas con 0 niños.\
Unos menos con 1 y 2; e incluso menos con 3. \
Sin embargo, nos figura una fila que reservo con 10 niños. Dicha fila es la siguiente:

```python
hotelsdf[hotelsdf["children_num"] == 10]
```

#### Ajustes de valor




Considerando que es un valor tanto mas alto que el resto, que es un unico caso y considerando que fue hecha con **2 adultos** nada mas; podemos considerar que este outlier y que lo podemos remover. 

```python
hotelsdf.drop((hotelsdf[hotelsdf["children_num"] == 10].index.values),inplace=True)
```

### days in the waiting list 


#### Valores estadisticos relevantes

```python
hotelsdf["days_in_waiting_list"].describe()
```

Days in waiting list representa la cantidad de dias que la reserva estuvo en la lista de espera antes de serconfirmada.
Esta variable es **discreta**, porque representa una cantidad discreta de dias.\
Esta esta alamacenada como int:

```python
print(hotelsdf["days_in_waiting_list"].dtype)
```

#### Valores nulos/faltantes

```python
hotelsdf.days_in_waiting_list.isna().sum()
```

No tiene valores vacios


#### Grafica de distribucion

```python
print("Los valores que toma la variable son los siguientes:")
daysInWaitingListValores = (hotelsdf["days_in_waiting_list"].unique())
daysInWaitingListValores.sort()
print(daysInWaitingListValores)
print()
print("Y toma dichos valores con la siguiente frecuencia")
hotelsdf["days_in_waiting_list"].value_counts()
```

```python
#plt.xlabel(xlabel = 'Dia de llegada')
#sns.boxplot(data = hotelsdf['days_in_waiting_list'])
#plt.title("Dia de llegada del mes")
#plt.ylabel(ylabel = 'Frecuencia')
#data = hotelsdf.days_in_waiting_list
#sns.kdeplot(data = data)
#plt.xlabel(xlabel = 'Average daily rate')
#plt.ylabel(ylabel = 'Frecuencia')
#plt.title('Distribucion del average daily rate')


#sns.boxplot(data = hotelsdf, x='days_in_waiting_list', palette='Set1')
```

#### Outliers


Los valores mas llamativos son aquellos por encima de 300; sin embargo no podemos establecer que son outliers porque son cantidades de dias


#### Ajustes de valor

Vamos a aplicar la tecnica de normalizar para poder aprovechar los datos. Podemos separarlo en 3 grandes grupos: Poco tiempo, mediano tiempo, mucho tiempo.\
Primero vamos a ver la cantidad de dias que hay en nuestro dataset


### lead time 


#### Valores estadisticos relevantes

```python
hotelsdf["lead_time"].describe()
```

Lead time representa la cantidad de dias que hubo entre el dia que se realizo la reserva y el dia de llegada.\
Esta variable es **discreta**, porque representa una cantidad discreta de dias.\
Esta esta alamacenada como int:

```python
print(hotelsdf["lead_time"].dtype)
```

#### Valores nulos/faltantes

```python
hotelsdf.days_in_waiting_list.isna().sum()
```

No tiene valores faltantes


#### Grafica de distribucion


Vamos a analizar la frecuencia de los distintos valores que lead time puede tomar

```python
hotelsdf["lead_time"].value_counts()
```

Vamos a graficarlos para ver su distribucion

```python
data = hotelsdf.lead_time
sns.kdeplot(data = data)
plt.xlabel(xlabel = 'Lead time')
plt.ylabel(ylabel = 'Frecuencia')
plt.title('Distribucion del lead time') #TODO: Cambiar la Y para ver la frecuencia, no esa numero raro
```

Vemos que la mayoria de los valores estan por debajo de 300

```python
leadTimeValores = (hotelsdf["lead_time"].unique())
leadTimeValores.sort()
print(leadTimeValores)
```

```python
sns.boxplot(data=hotelsdf.lead_time)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de noches de fin de semana")
plt.title("Canidad de noches de fin de semana por reserva")
plt.show()
```

```python
hotelsdf[hotelsdf["lead_time"] >= 400]
```

#### Outliers


```python
porcentaje = str((len(hotelsdf[hotelsdf["lead_time"] >= 400]) * 100)/len(hotelsdf))
print("Los valores mas llamativos son aquellos por encima de 400. Dichos valores representan un: " + porcentaje + "%")
```

Es un porcentaje lo suficientemente bajo para poder borrarlos

```python
hotelsdf.drop(hotelsdf[hotelsdf["lead_time"] >= 400].index, inplace = True)
hotelsdf.reset_index()
```

### previous booking not cancelled number

#### Valores estadisticos relevantes

```python
hotelsdf["previous_bookings_not_canceled_num"].describe()
```

Esta variable representa la cantidad de reservasa que no fueron canceladas por el usuario antes de la reserva actual


#### Valores nulos/faltantes

```python
hotelsdf.previous_bookings_not_canceled_num.isna().sum()
```

#### Grafica de distribucion


```python
eje_y = hotelsdf["previous_bookings_not_canceled_num"].value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de reservas no canceladas')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de reservas no canceladas')

hotelsdf["previous_bookings_not_canceled_num"].value_counts() #TODO: Corregir cuadro, se ve horrible el cuadro
```

#### Outliers
No parece haber ningun valor fuera  de lo comun


#### Ajustes de valor

Vamos a aplicar la tecnica de normalizar para poder aprovechar los datos. Podemos separarlo en 3 grandes grupos: Poco tiempo, mediano tiempo, mucho tiempo.\
Primero vamos a ver la cantidad de dias que hay en nuestro dataset


### previous booking cancellation number


#### Valores estadisticos relevantes
```python
hotelsdf["previous_cancellations_num"].describe()
```

```python
hotelsdf["previous_cancellations_num"].value_counts()
```

Esta variable representa la cantidad de reservasa que si fueron canceladas por el usuario antes de la reserva actual


#### Valores nulos/faltantes
```python
hotelsdf.previous_cancellations_num.isna().sum()
```

#### Grafica de distribucion
```python
hotelsdf["previous_cancellations_num"].value_counts() #TODO: Corregir cuadro, se ve horrible el cuadro
eje_y = hotelsdf["previous_cancellations_num"].value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(y = eje_y, x = eje_x, palette='Set2')
plt.xlabel('Cantidad de reservas canceladas')
plt.ylabel(ylabel='Frecuencia')
plt.title('Numero de reservas canceladas')

hotelsdf["previous_cancellations_num"].value_counts() #TODO: Corregir cuadro, se ve horrible el cuadro
```
#### Outliers
No parece haber ningun valor fuera  de lo comun


#### Ajustes de valor

### required car space number 

#### Valores estadisticos relevantes

```python
hotelsdf.required_car_parking_spaces_num.describe()
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.required_car_parking_spaces_num.isna().sum())
```

#### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='required_car_parking_spaces_num')
plt.title("Cantidad de reservas por espacios de estacionamiento")
```

#### Outliers


Viendo el grafico podemos identificar que el numero de espacios de estacionamiento mas comun es 0, seguido por 1. 
Además encontramos algunos pocos casos en los que se reservaron 2, 3 y 8 espacios.
Estos ultimos son posibles Outliers.

Mostramos dichos registros junto con la columnas de adult_num para analizarlos más en detalle y determinar si alguno de ellos puede ser Outlier y por que.
Nuestro criterio para determinar que un valor es adecuado para esta variable es que haya como mucho 1 espacio de estacionamiento por adulto en la reserva.

```python
registrosDosOMasEspacios = hotelsdf[hotelsdf["required_car_parking_spaces_num"]>=2]
#PREG deberia hacer un .copy x las dudas?
display(registrosDosOMasEspacios[['adult_num', "required_car_parking_spaces_num"]].sort_values(
    by = "required_car_parking_spaces_num", 
    ascending = False
))
```

De la tabla anterior se pueden sacar las siguientes conclusiones:
- El resgistro con valor de 8 espacios de estacionamiento es claramente un Outlier ya que no es coherente que una habitacion para dos personas haya reservado esa cantidad de espacios de estacionamiento.
- El resgistro con el valor de 3 espacios de estacionamiento y 2 adultos tambien es un Outliers ya que tampoco es coherente que 2 personas hayan reservado 3 espacios de estacionamiento.
- Los registros restantes NO son Outliers ya que si bien contienen valores poco freciuentes, son coherentes con el criterio explicado en el parrafo de arriba.


#### Ajustes de valor


Con el analisis anteior, tomamos las siguiuentes decisiones:
- Para el resgistro con valor de 8 espacios de estacionamiento,lo eliminamos por tratarse de un Outlier muy grosero.
- En el registro resgistro con el valor de 3 espacios de estacionamiento y 2 adultos, cambiamos el valor de required_car_parking_spaces_num por el valor "2" suponiendo un error de tipeo.
- Se mantienen sin cambios el resto de los registros restantes listados arriba.

```python
hotelsdf.loc[ (hotelsdf.required_car_parking_spaces_num==3) & (hotelsdf.adult_num==2) , "required_car_parking_spaces_num"] = 2
```

### special requests number 


#### Valores estadisticos relevantes

```python
hotelsdf.special_requests_num.describe() 
```

#### Valores nulos/faltanteS

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.special_requests_num.isna().sum())
```

#### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='special_requests_num', palette='Set1')
plt.title("Reservas por cantidad de requisitos especiales")
```

```python
sns.boxplot(data=hotelsdf.special_requests_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de requisitos especiales")
plt.title("Canidad de requisitos especiales por reserva")
plt.show()
```

#### Outliers


Viendo los graficos vemos que los valores mas frecuentes de requisitos especiales son 0 (ninguno), 1 y 2 y algunos menos con 3. Ademas hay muy pocos con 4 y 5. 
Los valores que podrian levantar sosppecha son 4 y 5.
Miramos la cantidad de registros de cada uno de ellos para ver que no sean casos puntuales.

```python
print("hay", hotelsdf[hotelsdf.special_requests_num==4].shape[0] ,"reservas con 4 requisitos especiales")
print("hay", hotelsdf[hotelsdf.special_requests_num==5].shape[0] ,"reservas con 5 requisitos especiales")
```

#### Ajustes de valor


Debido a la la cantidad de reservas para estos casos y que el rango de valores es relativamente acotado, no parcen ser casos puntuales. 
Procedemos a cambiar la cantidad de requisitos especiales de dichos registros el valor mas frecuente

```python
media_special_requests = round(hotelsdf.special_requests_num.mean())
hotelsdf.loc[hotelsdf['special_requests_num'] >= 4, 'special_requests_num'] = media_special_requests
```

### weekend nights number


#### Valores estadisticos relevantes

```python
hotelsdf.weekend_nights_num.describe() 
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.weekend_nights_num.isna().sum())
```

#### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='weekend_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de fin de semana")
```

```python
sns.boxplot(data=hotelsdf.weekend_nights_num)
plt.xlabel("Cantidad de reservas")
plt.ylabel("Canidad de noches de fin de semana")
plt.title("Canidad de noches de fin de semana por reserva")
plt.show()
```

#### Outliers


Podriamos suponer como posibles outliers, reservas con muchos dias de estadia. A simple vista se puede ver que hay pocas reservas con 5 o mas noches de fin de semana de estadia. Comenzamos estudiando los valores de 9 o mas dias de fin de semana ya que equivaldrian a un minimo de 4 semanas de estadia.

```python
mayores_a_nueve = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
mayores_a_nueve.shape[0]
```

```python
sns.countplot(data = mayores_a_nueve, x='weekend_nights_num', palette='Set1')
```

#### Ajustes de valor


Son solo 13 registros, es decir, representan muy poca cantidad del total. Tomamos la decision de eliminarlos para evitar que generen ruido al momento de generar el modelo.

```python
mas_de_nueve_noches_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=9]
hotelsdf.drop(mas_de_nueve_noches_finde.index, inplace = True)
hotelsdf.reset_index()
```

Hasta ahora analizamos las estadias con mas de 9 noches de fin de semana (al menos un mes de esatdia)
Sin embargo nos resta estudiar, los casos de 5, 6, 7 y 8 dias de fin de semana.
Vemos cuantos registros son

```python
mayores_a_5_menores_a_nueve_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=5]
mayores_a_5_menores_a_nueve_finde.shape[0]
```

```python
mayores_a_5_menores_a_nueve_finde = hotelsdf[hotelsdf["weekend_nights_num"]>=5]
sns.countplot(data = mayores_a_5_menores_a_nueve_finde, x='weekend_nights_num', palette='Set1')
```

Como son muchos mas registros posponemos su analisis para estudiarlos en un analisis multivariado despues de terminar de estudiar todas las variables cuantitativas pero los dejamos marcados como posibles registros a modificar.


### week nights number 


#### Valores estadisticos relevantes

```python
hotelsdf.week_nights_num.describe() 
```

#### Valores nulos/faltantes

```python
print("La cantidad de valores nulos/faltantes es", hotelsdf.week_nights_num.isna().sum())
```

#### Grafica de distribucion

```python
sns.countplot(data = hotelsdf, x='week_nights_num', palette='Set1')
plt.title("Reservas por cantidad de noches de de semana")
```

Se puede ver que la gran mayoria de las reservas son estadias de entre ningna (0) y 5 noches de semana y en menor medida estadias de entre 6 y 10 noches de semana. 
Como en el grafico no se ven puntualmente los registros con estadias de 11 o mas noches de semana, los graficamos de nuevo para ver como se distribuyen y estudiarlos mejor

```python
mayores_a_11_noches_semana = hotelsdf[hotelsdf["week_nights_num"]>=11]
mayores_a_11_noches_semana.shape[0]
```

```python
sns.countplot(data = mayores_a_11_noches_semana, x='week_nights_num', palette='Set1')
```

Como son muchos registros y no contienen valores incoherentes a primera vista posponemos su tratamiento para estudiarlos con un analisis multivariado comparandolo con weekend_nights_number en dicha seccion.


## Cualitativas

Variables cualitativas

En un principio establecemos una lista que contenga todas las variables cualitativas

```python
cualitativas = [
"agent_id",
"arrival_month",
"assigned_room_type",
"company_id",
"country",
"customer_type",
"deposit_type",
"distribution_channel",
"hotel_name",
'is_canceled',
"is_repeated_guest",
"market_segment_type",
"meal_type",
"reservation_status",
"reserved_room_type",
]
```

### Agent ID

#### Valores faltantes
EL CUADRO QUE VIENE TIENE QUE SER AJUSTADO PARA QUE SOLO MUESTRE COMPANY ID!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUIDADO!!!! RECORDAR CAMBIAR ANTES DE ENTREGAR
```python
cualitativas_nulas = hotelsdf[cualitativas].isnull().sum()
cualitativas_nulas = cualitativas_nulas[cualitativas_nulas > 0]

cuantitativas_nulas_per = pd.Series()

for indice in cualitativas_nulas.index:
    cuantitativas_nulas_per[indice] = cualitativas_nulas[indice]/len(hotelsdf[indice])*100

sns.barplot(x = cuantitativas_nulas_per.index, y = cuantitativas_nulas_per)
plt.ylabel(ylabel= 'Porcentaje')
plt.xlabel(xlabel= 'Nombre columna')
plt.title(label = 'Porcentaje de valores nulos')
plt.ylim(0, 100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.show()
```


#### Valores que toma

```python
agentIDValores = (hotelsdf["agent_id"].unique())
agentIDValores.sort()
print(agentIDValores)
```

#### Ajuste de valores faltantes

Reemplazamos valores faltantes por 0 ya que no existe previamente y servira para regular los tipos de atos de la columna

```python
hotelsdf.loc[hotelsdf['agent_id'].isnull(), 'agent_id'] = 0
hotelsdf[hotelsdf.agent_id.isnull()]
hotelsdf['agent_id'] = hotelsdf['agent_id'].astype(int)
```

#### Grafica de distribucion

```python
cantidad = len(hotelsdf['agent_id'].value_counts().index.tolist())
print(f"La cantidad de identificaciones de empresa es: {cantidad}")
```

Debido a que existen 295 id de empresas, graficamos un muestreo de los 10 ids mas frecuentes en el dataframe

```python
data = hotelsdf.agent_id.value_counts().sort_values(ascending=False).head(10)
sns.barplot(y = data, x = data.index.tolist())
#detallar 
```

#### Outliers
#### Ajustes de valor

### arrival month


#### Valores faltantes


#### Valores que toma


```python
arrivalMonthValores = (hotelsdf["arrival_month"].unique())
month_lookup = list(month_name)
months = arrivalMonthValores
sorted(months, key=month_lookup.index)
#print(leadTimeValores)
```

#### Grafica de distribucion

```python
eje_y = hotelsdf.arrival_month.value_counts()
eje_x = eje_y.index.tolist()
plt.figure(figsize=(8,5))
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```

#### Outliers
#### Ajustes de valor

### Assigned Room type


#### Valores faltantes



#### Valores que toma



```python
assigned_room_typeValores = (hotelsdf["assigned_room_type"].unique())
assigned_room_typeValores.sort()
print(assigned_room_typeValores)
```

```python
eje_y = hotelsdf.assigned_room_type.value_counts()
eje_x = eje_y.index.tolist()
plt.figure(figsize=(8,5))
sns.barplot(x = eje_x, y = eje_y)
```
#### Grafica de distribucion
#### Outliers
#### Ajustes de valor


### Company ID
#### Valores faltantes
EL CUADRO QUE VIENE TIENE QUE SER AJUSTADO PARA QUE SOLO MUESTRE COMPANY ID!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUIDADO!!!! RECORDAR CAMBIAR ANTES DE ENTREGAR
```python
cualitativas_nulas = hotelsdf[cualitativas].isnull().sum()
cualitativas_nulas = cualitativas_nulas[cualitativas_nulas > 0]

cuantitativas_nulas_per = pd.Series()

for indice in cualitativas_nulas.index:
    cuantitativas_nulas_per[indice] = cualitativas_nulas[indice]/len(hotelsdf[indice])*100

sns.barplot(x = cuantitativas_nulas_per.index, y = cuantitativas_nulas_per)
plt.ylabel(ylabel= 'Porcentaje')
plt.xlabel(xlabel= 'Nombre columna')
plt.title(label = 'Porcentaje de valores nulos')
plt.ylim(0, 100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.show()
```

#### Valores que toma
#### Grafica de distribucion
#### Outliers
#### Ajustes de valor

### Country


#### Valores faltantes
EL CUADRO QUE VIENE TIENE QUE SER AJUSTADO PARA QUE SOLO MUESTRE COMPANY ID!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
CUIDADO!!!! RECORDAR CAMBIAR ANTES DE ENTREGAR
```python
cualitativas_nulas = hotelsdf[cualitativas].isnull().sum()
cualitativas_nulas = cualitativas_nulas[cualitativas_nulas > 0]

cuantitativas_nulas_per = pd.Series()

for indice in cualitativas_nulas.index:
    cuantitativas_nulas_per[indice] = cualitativas_nulas[indice]/len(hotelsdf[indice])*100

sns.barplot(x = cuantitativas_nulas_per.index, y = cuantitativas_nulas_per)
plt.ylabel(ylabel= 'Porcentaje')
plt.xlabel(xlabel= 'Nombre columna')
plt.title(label = 'Porcentaje de valores nulos')
plt.ylim(0, 100)
plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
plt.show()
```


#### Valores que toma


```python
countryValores = (hotelsdf["country"].unique())
#countryValores.sort() #No se puede ordenar porque tiene valores nan
print(countryValores)
```

#### Grafica de distribucion
```python
data = hotelsdf.country.value_counts().sort_values(ascending=False).head(20)
plt.xticks(rotation=45)
sns.barplot(y = data, x = data.index.tolist())
```
#### Ajuste de valores faltantes
#### Outliers
#### Ajustes de valor

### Custemer type
#### Valores faltantes


#### Valores que toma


```python
customer_typeValores = (hotelsdf["customer_type"].unique())
customer_typeValores.sort()
print(customer_typeValores)
```

#### Grafica de distribucion

```python
eje_y = hotelsdf.customer_type.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```

#### Outliers
#### Ajustes de valor

### Deposit type
#### Valores faltantes


#### Valores que toma


```python
deposit_typeValores = (hotelsdf["deposit_type"].unique())
deposit_typeValores.sort()
print(deposit_typeValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.deposit_type.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Distribution channel
#### Valores faltantes


#### Valores que toma


```python
distribution_channelValores = (hotelsdf["distribution_channel"].unique())
distribution_channelValores.sort()
print(distribution_channelValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.distribution_channel.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Hotel Name
#### Valores faltantes


#### Valores que toma


```python
hotel_nameValores = (hotelsdf["hotel_name"].unique())
hotel_nameValores.sort()
print(hotel_nameValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.hotel_name.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Is canceled (Target)
#### Valores faltantes


#### Valores que toma


```python
is_canceledValores = (hotelsdf["is_canceled"].unique())
is_canceledValores.sort()
print(is_canceledValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.is_canceled.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Is repeated guest
#### Valores faltantes


#### Valores que toma


```python
is_repeated_guestValores = (hotelsdf["is_repeated_guest"].unique())
is_repeated_guestValores.sort()
print(is_repeated_guestValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.is_repeated_guest.value_counts()
eje_x = eje_y.index.tolist()
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor


### Market segment
#### Valores faltantes


#### Valores que toma


```python
market_segment_typeValores = (hotelsdf["market_segment_type"].unique())
market_segment_typeValores.sort()
print(market_segment_typeValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.market_segment_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### meal type
#### Valores faltantes


#### Valores que toma


```python
meal_typeValores = (hotelsdf["meal_type"].unique())
meal_typeValores.sort()
print(meal_typeValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.meal_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Reservation Status
#### Valores faltantes


#### Valores que toma


```python
reservation_statusValores = (hotelsdf["reservation_status"].unique())
reservation_statusValores.sort()
print(reservation_statusValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.reservation_status.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

### Reserved room type
#### Valores faltantes


#### Valores que toma


```python
reserved_room_typeValores = (hotelsdf["reserved_room_type"].unique())
reserved_room_typeValores.sort()
print(reserved_room_typeValores)
```

#### Grafica de distribucion
```python
eje_y = hotelsdf.reserved_room_type.value_counts()
eje_x = eje_y.index.tolist()
plt.xticks(rotation=45)
sns.barplot(x = eje_x, y = eje_y)
```
#### Outliers
#### Ajustes de valor

Como ya habiamos observado en la cantidad de dias de fin de semana, la mayor cantidad de gente se queda 



# Analisis multivariado
## Medicion de la correlacion entre las variables cuantitativas

Una vez hecho el tratado sobre outliers y datos faltantes medimos la correlacion entre las variables cuantitativas encontradas en el dataframe

```python
# Este if es se usa para evitar problemas de versiones de pandas entre la version local y la presente en Google Collab
if (pd.__version__) == "1.5.2":
    correlaciones = hotelsdf[cuantitativas].corr(numeric_only=True)

else:
    correlaciones = hotelsdf[cuantitativas].corr()

    
sns.set(style = 'darkgrid')
plt.figure( figsize = (12, 9))
sns.heatmap(data = correlaciones,annot = True, vmin = -1, vmax =1, fmt='.2f')
sns.color_palette("mako", as_cmap=True)
plt.show()
```

### (lo de abasjo va a multivariadoooo)

Como dijimos previamente, analizar por separado las noches de dia de semana y las noches de dias de fin de semana no basta parta estudiar dichas variables. El primer problema que podria surgir es que la cantidad de noches de semana y de fin de semana no guarden una relacion coherente. Al graficarlos debería ocurrir lo siguiente:
- Cuando el numero de noches n de fin de semana es impar, las pendientes de las rectas tienen una variacion de +/- 5 noches de semana
- Cuando el numero de noches n de fin de semana es par, las pendientes de las rectas tienen una variacion de +- 10 noches de semana

```python
sns.scatterplot(x=hotelsdf.weekend_nights_num,y=hotelsdf.week_nights_num)
plt.title('Dispersograma noches finde vs noches de semana')
plt.show()
```

Al observar el grafico observamos que todos los puntos se encuentran en los rangos explicados anteriormente.


Sin embargo, ocurre que si bien puede resultar util tener datos sobre las noches de semana y las de fin de semana, un dato que nos podria resultar aun mas util es la cantidad de noches totales de estadia.
Agregamos una columna con dicho dato

```python
hotelsdf["dias_totales"] = hotelsdf["week_nights_num"] + hotelsdf["weekend_nights_num"]
```

Puesto que ahora tenemos una nueva variable, realizamos un breve analisis univariado sobre la misma para determinar si existen Outliers no detectados en las columans de week y weekend nights number.

```python
plt.figure(figsize=(15,5))
sns.countplot(data = hotelsdf, x = 'dias_totales', palette= 'Set2')
plt.title('Cantidad de reservas por dias de estadia')
plt.xlabel('Dias de estadia')
plt.ylabel('Frecuencia')
```

La mayoria de las reservas son de estadias de entre 1 y 7 dias de estadia. En menor medida se observan reservas para estadias entre 8 y 14 dias y por ultimos unas pocas entre 15 y 30 dias. Realizamos un boxplot para darnos una idea de que numero utilizar como corte para determinar outliers.

```python
plt.xlabel(xlabel = 'Dia estadia')
sns.boxplot(data = hotelsdf['dias_totales'])
plt.title("reservas por dias de estadia")
plt.ylabel(ylabel = 'Frecuencia')
```

Segun el grafico se alejarian de la media todos los valores de 8 o mas dias de estadia. Vemos cuantos registros son y que porcentaje representan del total

```python
reservas_mas_de_ocho_dias = hotelsdf[(hotelsdf.dias_totales>=8)].shape[0]
print("hay",reservas_mas_de_ocho_dias,"que representan un porcentaje del total de", reservas_mas_de_ocho_dias*100/hotelsdf.shape[0],"%")
```

Puesto que este valor es muy elevado, apelamos al sentido comun. Rservas de hasta 14 dias de estadia son muy posibles, por lo cual estuidiamos las de mas 15 o mas dias.

```python
quince_o_mas_dias = hotelsdf[hotelsdf["dias_totales"]>=15]
#plt.figure(figsize=(15,5))
sns.countplot(data = quince_o_mas_dias, x = 'dias_totales', palette= 'Set2', hue = "is_canceled")
plt.title('Cantidad de dias de estadia')
plt.xlabel('Dias de estadia')
plt.ylabel('Frecuencia')
```

Primero vemos cuantos registros son en total

```python
print("hay",quince_o_mas_dias.shape[0],"que se quedan 15 o mas dias y representan un porcentaje del total de", (quince_o_mas_dias.shape[0])*100/hotelsdf.shape[0],"%")
```

Luego vemos cuantos de esos cancelan

```python
cancelaron_y_quince_o_mas_dias = hotelsdf[ (hotelsdf.dias_totales>=15) & (hotelsdf.is_canceled == 1) ].shape[0]

print("hay",cancelaron_y_quince_o_mas,"que cancelaron y se quedaron mas de 15 o mas dias.Osea un", cancelaron_y_quince_o_mas_dias*100/quince_o_mas_dias.shape[0],"% de los que se que se quedan mas de 15 dias cancelan")
```

Vemos que el porcentaje de reservas de mas de 15 dias que cancelan es muy alto. Sin embargo, la cantidad de registros con los que ocurre esto son muy pocos. Dejarlos, podria generar ruido al momento de realizar la prediccion. Nos podria llevar, erroneamente a pensar que alguien que se quedo muchos dias cancelaria cuando esto no necesariamente es asi. En el problema q estamos resolviendo, es prefereible no detectar a alguien que cancela, que suponer que alguien cancelaria y que luego no lo haga ya que en terminos de presupuestos, disponibilidad o cualquiera sea el uso que se le de a esta prediccion, no estar preparardo para una reserva perjudicaria mucho mas que estarlo "por las dudas".
Procedemos a eliminarlos

```python
a_eliminar_con_quince_o_mas_dias = hotelsdf[hotelsdf['dias_totales'] >= 15]
hotelsdf.drop(a_eliminar_con_quince_o_mas_dias.index, inplace = True)
hotelsdf.reset_index()
```

Intentamos hacer un calculo de distancia Mahalanobis pero debido a la cantidad de datos, nos generaba errores, por ello lo dejamos comentado.

"MemoryError: Unable to allocate 26.7 GiB for an array with shape (59903, 59903) and data type float64"

```python
# #Calulo el vector de medias
# vmedias=np.mean(hotelsdf[['weekend_nights_num','week_nights_num']])

# #Calculo la diferencia entre las observaciones y el vector de medias
# weekend_nights_dif = hotelsdf[['weekend_nights_num','week_nights_num']] - vmedias

# #Calculo matriz de covarianza y su inversa
# cov=hotelsdf[['weekend_nights_num','week_nights_num']].cov().values
# inv_cov = np.linalg.inv(cov)

# #Calculamos el cuadrado de la distancia de mahalanobis
# mahal =np.dot( np.dot(weekend_nights_dif, inv_cov) , weekend_nights_dif.T)

# hotelsdf['mahal_week_weekend_nights']=mahal.diagonal()
```

```python
sns.countplot(data = hotelsdf, x='dias_totales', hue='is_canceled')
```

Nos dio lo esperado. No hay datos incosistentes en cuanto a su comparacion con el numero de noches de semana.


Anslisis de Mahalanobis



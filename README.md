# Regressão Linear Multipla

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
```


```python
pd.set_option("float_format", "{:.2f}".format) # Configurando para retirar a notação cientifica

bd = pd.read_csv("data.csv")
bd
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>street</th>
      <th>city</th>
      <th>statezip</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-05-02 00:00:00</td>
      <td>313000.00</td>
      <td>3.00</td>
      <td>1.50</td>
      <td>1340</td>
      <td>7912</td>
      <td>1.50</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1340</td>
      <td>0</td>
      <td>1955</td>
      <td>2005</td>
      <td>18810 Densmore Ave N</td>
      <td>Shoreline</td>
      <td>WA 98133</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-02 00:00:00</td>
      <td>2384000.00</td>
      <td>5.00</td>
      <td>2.50</td>
      <td>3650</td>
      <td>9050</td>
      <td>2.00</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>3370</td>
      <td>280</td>
      <td>1921</td>
      <td>0</td>
      <td>709 W Blaine St</td>
      <td>Seattle</td>
      <td>WA 98119</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-02 00:00:00</td>
      <td>342000.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>1930</td>
      <td>11947</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1930</td>
      <td>0</td>
      <td>1966</td>
      <td>0</td>
      <td>26206-26214 143rd Ave SE</td>
      <td>Kent</td>
      <td>WA 98042</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-02 00:00:00</td>
      <td>420000.00</td>
      <td>3.00</td>
      <td>2.25</td>
      <td>2000</td>
      <td>8030</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1000</td>
      <td>1000</td>
      <td>1963</td>
      <td>0</td>
      <td>857 170th Pl NE</td>
      <td>Bellevue</td>
      <td>WA 98008</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-02 00:00:00</td>
      <td>550000.00</td>
      <td>4.00</td>
      <td>2.50</td>
      <td>1940</td>
      <td>10500</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1140</td>
      <td>800</td>
      <td>1976</td>
      <td>1992</td>
      <td>9105 170th Ave NE</td>
      <td>Redmond</td>
      <td>WA 98052</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4595</th>
      <td>2014-07-09 00:00:00</td>
      <td>308166.67</td>
      <td>3.00</td>
      <td>1.75</td>
      <td>1510</td>
      <td>6360</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1510</td>
      <td>0</td>
      <td>1954</td>
      <td>1979</td>
      <td>501 N 143rd St</td>
      <td>Seattle</td>
      <td>WA 98133</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4596</th>
      <td>2014-07-09 00:00:00</td>
      <td>534333.33</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>1460</td>
      <td>7573</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1460</td>
      <td>0</td>
      <td>1983</td>
      <td>2009</td>
      <td>14855 SE 10th Pl</td>
      <td>Bellevue</td>
      <td>WA 98007</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4597</th>
      <td>2014-07-09 00:00:00</td>
      <td>416904.17</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>3010</td>
      <td>7014</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3010</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>759 Ilwaco Pl NE</td>
      <td>Renton</td>
      <td>WA 98059</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4598</th>
      <td>2014-07-10 00:00:00</td>
      <td>203400.00</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>2090</td>
      <td>6630</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1070</td>
      <td>1020</td>
      <td>1974</td>
      <td>0</td>
      <td>5148 S Creston St</td>
      <td>Seattle</td>
      <td>WA 98178</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4599</th>
      <td>2014-07-10 00:00:00</td>
      <td>220600.00</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>1490</td>
      <td>8102</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1490</td>
      <td>0</td>
      <td>1990</td>
      <td>0</td>
      <td>18717 SE 258th St</td>
      <td>Covington</td>
      <td>WA 98042</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
<p>4600 rows × 18 columns</p>
</div>



# Separando a base

Lembrando que se deve fazer uma análise mais completa, mas irei direto ao modelo


```python
# Aqui um tratamento para eliminar linhas onde o preço = 0
bd = bd.drop(bd.loc[bd['price'] <= 0].index)
bd
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>condition</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>street</th>
      <th>city</th>
      <th>statezip</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-05-02 00:00:00</td>
      <td>313000.00</td>
      <td>3.00</td>
      <td>1.50</td>
      <td>1340</td>
      <td>7912</td>
      <td>1.50</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1340</td>
      <td>0</td>
      <td>1955</td>
      <td>2005</td>
      <td>18810 Densmore Ave N</td>
      <td>Shoreline</td>
      <td>WA 98133</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-05-02 00:00:00</td>
      <td>2384000.00</td>
      <td>5.00</td>
      <td>2.50</td>
      <td>3650</td>
      <td>9050</td>
      <td>2.00</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>3370</td>
      <td>280</td>
      <td>1921</td>
      <td>0</td>
      <td>709 W Blaine St</td>
      <td>Seattle</td>
      <td>WA 98119</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-05-02 00:00:00</td>
      <td>342000.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>1930</td>
      <td>11947</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1930</td>
      <td>0</td>
      <td>1966</td>
      <td>0</td>
      <td>26206-26214 143rd Ave SE</td>
      <td>Kent</td>
      <td>WA 98042</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2014-05-02 00:00:00</td>
      <td>420000.00</td>
      <td>3.00</td>
      <td>2.25</td>
      <td>2000</td>
      <td>8030</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1000</td>
      <td>1000</td>
      <td>1963</td>
      <td>0</td>
      <td>857 170th Pl NE</td>
      <td>Bellevue</td>
      <td>WA 98008</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014-05-02 00:00:00</td>
      <td>550000.00</td>
      <td>4.00</td>
      <td>2.50</td>
      <td>1940</td>
      <td>10500</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1140</td>
      <td>800</td>
      <td>1976</td>
      <td>1992</td>
      <td>9105 170th Ave NE</td>
      <td>Redmond</td>
      <td>WA 98052</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4595</th>
      <td>2014-07-09 00:00:00</td>
      <td>308166.67</td>
      <td>3.00</td>
      <td>1.75</td>
      <td>1510</td>
      <td>6360</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1510</td>
      <td>0</td>
      <td>1954</td>
      <td>1979</td>
      <td>501 N 143rd St</td>
      <td>Seattle</td>
      <td>WA 98133</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4596</th>
      <td>2014-07-09 00:00:00</td>
      <td>534333.33</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>1460</td>
      <td>7573</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1460</td>
      <td>0</td>
      <td>1983</td>
      <td>2009</td>
      <td>14855 SE 10th Pl</td>
      <td>Bellevue</td>
      <td>WA 98007</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4597</th>
      <td>2014-07-09 00:00:00</td>
      <td>416904.17</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>3010</td>
      <td>7014</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>3010</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>759 Ilwaco Pl NE</td>
      <td>Renton</td>
      <td>WA 98059</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4598</th>
      <td>2014-07-10 00:00:00</td>
      <td>203400.00</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>2090</td>
      <td>6630</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1070</td>
      <td>1020</td>
      <td>1974</td>
      <td>0</td>
      <td>5148 S Creston St</td>
      <td>Seattle</td>
      <td>WA 98178</td>
      <td>USA</td>
    </tr>
    <tr>
      <th>4599</th>
      <td>2014-07-10 00:00:00</td>
      <td>220600.00</td>
      <td>3.00</td>
      <td>2.50</td>
      <td>1490</td>
      <td>8102</td>
      <td>2.00</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1490</td>
      <td>0</td>
      <td>1990</td>
      <td>0</td>
      <td>18717 SE 258th St</td>
      <td>Covington</td>
      <td>WA 98042</td>
      <td>USA</td>
    </tr>
  </tbody>
</table>
<p>4551 rows × 18 columns</p>
</div>




```python
# Separando os valores de X e Y
x = bd.iloc[:, 2: 14].values
y = bd.iloc[:, 1].values

```


```python
#Criando a base de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.15, random_state=43)

x_treino.shape, x_teste.shape
```




    ((3868, 12), (683, 12))




```python
# Agora vamos treinar o modelo
regressor = LinearRegression()
regressor.fit(x_treino, y_treino)

```


```python
# Modelo Treinado, agora verificamos o Score
regressor.score(x_teste, y_teste)
```




    0.6005745884578182



# Agora é observar o resultado do modelo. Estarei utilizando a base de testes para comparar os resultados


```python
previsao = regressor.predict(x_teste)
previsao[20]
```




    545300.7472258131




```python
y_teste[20]
```




    566000.0



* Veja que a previsão foi de 545300,74 enquanto o valor real era 566000,00


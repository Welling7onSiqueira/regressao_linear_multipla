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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" checked><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">LinearRegression</label><div class="sk-toggleable__content"><pre>LinearRegression()</pre></div></div></div></div></div>




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


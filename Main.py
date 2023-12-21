import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# FASE II
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.manifold import MDS

# FASE III
from scipy.stats import yeojohnson
from scipy.stats import boxcox

# FASE IV y FASE V
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from SVM import support_vector_machine
from sklearn.linear_model import LogisticRegression

#%% FASE II. COMPRENSIÓN DE LOS DATOS

#%%% Fuente de datos
# Recopilación de los datos
df = pd.read_csv("loan_data.csv")

#%%% Exploración de los datos
#%%%% Información de las variables
class_distribution = df['not.fully.paid'].value_counts() #verificar si está balanceado
print(class_distribution)

print("\n\nDescribe: \n",df.describe()) #estadísticos básicos
print("\n\n NaN Values: \n",df.isna().sum()) #Valores nulos
print("\n\nInfo:\n",df.info) #Información de dataframe
print("\n\nTipos:\n",df.dtypes) #Tipos de datos
print("\n\nValores únicos:\n",df.nunique()) #valores únicos

#%%%% Histogramas

# Se seleccionan las columnas numéricas.
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col], bins=10)  
    plt.title(f'Histograma de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()

#%%%% Countplots
# Se seleccionan las columnas categóricas
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x=col)
    plt.title(f'Gráfico de Conteo para {col}')
    plt.xlabel(col)
    plt.ylabel('Conteo')
    plt.xticks(rotation=90)  
    plt.show()

#%%%% Gráficas de caja
for col in num_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f'Diagrama de Caja de {col}')
    plt.show()

    
#%%% Relaciones entre variables (gráfica de dispersión)

#%%%% todas
# Especificar los colores para la variable objetivo
colors = {0: 'green', 1: 'red'}

# Añadir una columna de colores al DataFrame
df['color'] = df['not.fully.paid'].map(colors)

# Seleccionar solo las variables numéricas (excluyendo la variable objetivo y de colores)
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Crear un pairplot con colores diferenciados por la variable objetivo
sns.pairplot(df, hue='not.fully.paid', palette=colors, vars=numeric_columns)
plt.suptitle('Pairplot de Variables con Colores por Variable Objetivo', y=1.02)

# Mostrar el gráfico
plt.show()

#%%%% seleccionadas

# Variables seleccionadas
selected_vars = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths']

# Crear un DataFrame con las variables seleccionadas y la variable objetivo
df_selected = df[selected_vars + ['not.fully.paid']]

# Asignar colores a la variable objetivo
colors = {0: 'green', 1: 'red'}
df_selected['colors'] = df_selected['not.fully.paid'].map(colors)

# Visualización con pairplot
sns.pairplot(df_selected, hue='not.fully.paid', palette=colors)
plt.show()

#%%% Relaciones entre variables (mapa de calor)

# Calcular las matrices de correlación
corr_pearson = df.corr(method='pearson')
corr_spearman = df.corr(method='spearman')
corr_kendall = df.corr(method='kendall')

# Configurar el tamaño de la figura
plt.figure(figsize=(15, 5))

# Mapa de calor para la correlación de Pearson
plt.subplot(1, 3, 1)
sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Pearson')

# Mapa de calor para la correlación de Spearman
plt.subplot(1, 3, 2)
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Spearman')

# Mapa de calor para la correlación de Kendall
plt.subplot(1, 3, 3)
sns.heatmap(corr_kendall, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlación de Kendall')

# Ajustar el diseño
plt.tight_layout()

# Mostrar el gráfico
plt.show()

#%%% Visualización con reducción de dimensionalidad (PCA a 3D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid','color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar PCA a 3 dimensiones
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales y la variable objetivo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['not.fully.paid'] = y

# Visualizar en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = ax.scatter(
    xs=df_pca['PC1'],
    ys=df_pca['PC2'],
    zs=df_pca['PC3'],
    c=df_pca['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('PCA 3 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
ax.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%%% Visualización con reducción de dimensionalidad (PCA a 2D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid','color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar PCA a 2 dimensiones
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Crear un DataFrame con las componentes principales y la variable objetivo
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['not.fully.paid'] = y

# Visualizar en 2D
plt.figure(figsize=(10, 8))

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = plt.scatter(
    x=df_pca['PC1'],
    y=df_pca['PC2'],
    c=df_pca['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
plt.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%%% Visualización con reducción de dimensionalidad (MDS a 3D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid', 'color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para MDS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar MDS a 3 dimensiones
mds = MDS(n_components=3)
X_mds = mds.fit_transform(X_scaled)

# Crear un DataFrame con las dimensiones MDS y la variable objetivo
df_mds = pd.DataFrame(X_mds, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
df_mds['not.fully.paid'] = y

# Visualizar en 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = ax.scatter(
    xs=df_mds['Dimension 1'],
    ys=df_mds['Dimension 2'],
    zs=df_mds['Dimension 3'],
    c=df_mds['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
ax.set_title('MDS 3 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
ax.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()

#%%% Visualización con reducción de dimensionalidad (MDS a 2D)

# Separar las características (X) y la variable objetivo (y)
X = df.drop(['not.fully.paid', 'color'], axis=1)
y = df['not.fully.paid']

# Utilizar Binary Encoder para manejar variables categóricas
encoder = ce.BinaryEncoder(cols=['purpose'])
X_encoded = encoder.fit_transform(X)

# Estandarizar las características para MDS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Aplicar MDS a 2 dimensiones
mds = MDS(n_components=2)
X_mds = mds.fit_transform(X_scaled)

# Crear un DataFrame con las dimensiones MDS y la variable objetivo
df_mds = pd.DataFrame(X_mds, columns=['Dimension 1', 'Dimension 2'])
df_mds['not.fully.paid'] = y

# Visualizar en 2D
plt.figure(figsize=(10, 8))

# Scatter plot para ambas clases con colores personalizados
colors = {0: 'green', 1: 'red'}
scatter = plt.scatter(
    x=df_mds['Dimension 1'],
    y=df_mds['Dimension 2'],
    c=df_mds['not.fully.paid'].map(colors),
    marker='o'
)

# Configuraciones adicionales para mejorar la visualización
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('MDS 2 Dimensiones')

# Mostrar la leyenda
legend_labels = {0: 'Fully Paid (0)', 1: 'Not Fully Paid (1)'}
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[label], markersize=10, label=legend_labels[label]) for label in legend_labels]
plt.legend(handles=handles, title='Variable Objetivo', loc='upper right')

plt.show()


#%% FASE III. TRANSFORMACIÓN DE LOS DATOS

#%%% Codificación de purpose (Binary Encoder)

# Inicializar el codificador BinaryEncoder
encoder = ce.BinaryEncoder(cols=['purpose'])

# Aplicar la codificación al DataFrame
df_encoded = encoder.fit_transform(df)

# Ver el DataFrame resultante
print(df_encoded.head())

#%%% Manejo de valores atípicos

columns_of_interest = ['installment', 'log.annual.inc', 'fico', 'days.with.cr.line', 'revol.bal', 'inq.last.6mths']

# Calcula el Z-Score para cada columna de interés
for column in columns_of_interest:
    z_score_column = column + '_Z_Score'
    df_encoded[z_score_column] = (df_encoded[column] - df_encoded[column].mean()) / df_encoded[column].std()

# Define el umbral para identificar valores atípicos
umbral = 2

# Identifica registros con valores atípicos en al menos una de las columnas
outliers_df = df_encoded[(df_encoded.filter(regex='_Z_Score').abs() > umbral).any(axis=1)]

# Muestra el DataFrame con valores atípicos
print(outliers_df)

# Elimina los registros en df_encoded que están en outliers_df
indices_to_drop = outliers_df.index
df_encoded = df_encoded.drop(indices_to_drop)


# Lista de las columnas a eliminar
columns_to_drop = [column + '_Z_Score' for column in columns_of_interest]

# Eliminar las columnas
df_encoded = df_encoded.drop(columns=columns_to_drop)

#df_encoded = df_encoded.drop(['color'], axis=1)

#%%% Normalización de los datos.

# Selecciona todas las columnas numéricas para aplicar la normalización
#numeric_columns = df_encoded.select_dtypes(include=['float64', 'int64']).columns
numeric_columns = [ 'int.rate', 'installment', 'log.annual.inc', 'dti',
                   'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']

# Inicializa el objeto StandardScaler
scaler = StandardScaler()

# Aplica la normalización Z-Score a todas las columnas seleccionadas
df_encoded[numeric_columns] = scaler.fit_transform(df_encoded[numeric_columns])

# Muestra el DataFrame resultante con las columnas normalizadas
print(df_encoded)

#df_encoded.to_csv("data.csv", index=False)

#%%% Transformaciones para aproximar a distribución normal (Yeo-Johnson, Box-Cox)

#df_encoded = pd.read_csv("data.csv")

columns_to_transform = ['days.with.cr.line', 'revol.bal']

for col in columns_to_transform:
    plt.figure(figsize=(8, 6))
    plt.hist(df_encoded[col], bins=10)  
    plt.title(f'Histograma de {col} (antes)')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()
 

df_encoded['days.with.cr.line'], _ = yeojohnson(df_encoded['days.with.cr.line'] + 1)  # Agregamos 1 para manejar valores no positivos
df_encoded['revol.bal'], _ = boxcox(df_encoded['revol.bal'] + 1)  # Agregamos 1 para manejar valores no positivos

for col in columns_to_transform:
    plt.figure(figsize=(8, 6))
    plt.hist(df_encoded[col], bins=10)  
    plt.title(f'Histograma de {col} (después)')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    plt.show()
    
#df_encoded.to_csv("data.csv", index=False)

#%% FASE IV. MODELADO DE DATOS: HOLDOUT

df = pd.read_csv("data.csv") 

#%%% Random Forest

# Selecciona las características (X) y la variable objetivo (y)
#X = df.drop('not.fully.paid', axis=1)
X = df[['credit.policy', 'purpose_0', 'purpose_1', 'purpose_2', 'int.rate',
        'fico', 'days.with.cr.line',
       'revol.bal', 'revol.util', 'inq.last.6mths' ]]
y = df['not.fully.paid']

# Divide los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998)

# Inicializa el modelo de Random Forest
random_forest_model = RandomForestClassifier(random_state=1998)

# Entrena el modelo en el conjunto de entrenamiento
random_forest_model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = random_forest_model.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_rfh = accuracy_score(y_test, y_pred)
conf_matrix_rfh = confusion_matrix(y_test, y_pred)
class_report_rfh = classification_report(y_test, y_pred)

#%%% SVM Lineal

svm = support_vector_machine()
X_train = X_train.values  # Convertir DataFrame a un array de NumPy
X_test = X_test.values  # Convertir DataFrame a un array de NumPy

w,b = svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_svmh = accuracy_score(y_test, y_pred)
conf_matrix_svmh = confusion_matrix(y_test, y_pred)
class_report_svmh = classification_report(y_test, y_pred)

#%%% Decision Tree

# Selecciona las características (X) y la variable objetivo (y)
#X = df.drop('not.fully.paid', axis=1)
X = df[['credit.policy', 'purpose_0', 'purpose_1', 'purpose_2', 'int.rate',
        'fico', 'days.with.cr.line',
       'revol.bal', 'revol.util', 'inq.last.6mths' ]]
y = df['not.fully.paid']

# Divide los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998)

# Inicializa el clasificador de árbol de decisión
decision_tree = DecisionTreeClassifier(random_state=1998)

# Entrena el árbol de decisión con los datos de entrenamiento
decision_tree.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = decision_tree.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_dth = accuracy_score(y_test, y_pred)
conf_matrix_dth = confusion_matrix(y_test, y_pred)
class_report_dth = classification_report(y_test, y_pred)

#%%% Logistic Regression

# Selecciona las características (X) y la variable objetivo (y)
#X = df.drop('not.fully.paid', axis=1)
X = df[['credit.policy', 'purpose_0', 'purpose_1', 'purpose_2', 'int.rate',
        'fico', 'days.with.cr.line',
       'revol.bal', 'revol.util', 'inq.last.6mths' ]]
y = df['not.fully.paid']

# Divide los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998)


# Inicializa el clasificador de regresión logística
logistic_regression = LogisticRegression(random_state=1998)

# Entrena el modelo con los datos de entrenamiento
logistic_regression.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = logistic_regression.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_lrh = accuracy_score(y_test, y_pred)
conf_matrix_lrh = confusion_matrix(y_test, y_pred)
class_report_lrh = classification_report(y_test, y_pred)
#%% FASE V. EVALUACIÓN (Holdout)

# Imprime las métricas de evaluación Random Forest (Holdout)
print('_'*75)
print("\n\n Random Forest (Holdout)\n")
print(f'Accuracy: {accuracy_rfh:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_rfh)
print('\nClassification Report:')
print(class_report_rfh)

# Imprime las métricas de evaluación SVM (Holdout)
print('_'*75)
print("\n\n SVM (Holdout)\n")
print(f'Accuracy: {accuracy_svmh:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_svmh)
print('\nClassification Report:')
print(class_report_svmh)

# Imprime las métricas de evaluación Decision Tree (Holdout)
print('_'*75)
print("\n\n Decision Tree (Holdout)\n")
print(f'Accuracy: {accuracy_dth:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_dth)
print('\nClassification Report:')
print(class_report_dth)

# Imprime las métricas de evaluación Logistic Regression (Holdout)
print('_'*75)
print("\n\n Logistic Regression (Holdout)\n")
print(f'Accuracy: {accuracy_lrh:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_lrh)
print('\nClassification Report:')
print(class_report_lrh)

#%% FASE IV. MODELADO DE DATOS: VALIDACIÓN ESTRATIFICADA

df = pd.read_csv("data.csv") 

#%%% Random Forest

# Selecciona las características (X) y la variable objetivo (y)
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

# Divide los datos de manera estratificada en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1998, stratify=y)

# Inicializa el modelo de Random Forest
random_forest_model = RandomForestClassifier(random_state=1998)

# Entrena el modelo en el conjunto de entrenamiento
random_forest_model.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = random_forest_model.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_rfs = accuracy_score(y_test, y_pred)
conf_matrix_rfs = confusion_matrix(y_test, y_pred)
class_report_rfs = classification_report(y_test, y_pred)


#%%% SVM Lineal

svm = support_vector_machine()
X_train = X_train.values  # Convertir DataFrame a un array de NumPy
X_test = X_test.values  # Convertir DataFrame a un array de NumPy

w,b = svm.fit(X_train,y_train)

y_pred = svm.predict(X_test)
# Evalúa el rendimiento del modelo
accuracy_svms = accuracy_score(y_test, y_pred)
conf_matrix_svms = confusion_matrix(y_test, y_pred)
class_report_svms = classification_report(y_test, y_pred)

#%%% Decision Tree

# Selecciona las características (X) y la variable objetivo (y)
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

# Divide los datos de manera estratificada en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# Inicializa el clasificador de árbol de decisión
decision_tree = DecisionTreeClassifier(random_state=1998)

# Entrena el árbol de decisión con los datos de entrenamiento
decision_tree.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = decision_tree.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_dts = accuracy_score(y_test, y_pred)
conf_matrix_dts = confusion_matrix(y_test, y_pred)
class_report_dts = classification_report(y_test, y_pred)


#%%% Logistic Regression

# Selecciona las características (X) y la variable objetivo (y)
X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']

# Divide los datos de manera estratificada en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3, stratify=y)

# Inicializa el clasificador de regresión logística
logistic_regression = LogisticRegression(random_state=1998)

# Entrena el modelo con los datos de entrenamiento
logistic_regression.fit(X_train, y_train)

# Realiza predicciones en el conjunto de prueba
y_pred = logistic_regression.predict(X_test)

# Evalúa el rendimiento del modelo
accuracy_lrs = accuracy_score(y_test, y_pred)
conf_matrix_lrs = confusion_matrix(y_test, y_pred)
class_report_lrs = classification_report(y_test, y_pred)

#%% FASE V. EVALUACIÓN (validación estratificada)

# Imprime las métricas de evaluación Random Forest (Stratify)
print('_'*75)
print("\n\n Random Forest (Stratify)\n")
print(f'Accuracy: {accuracy_rfs:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_rfs)
print('\nClassification Report:')
print(class_report_rfs)

# Imprime las métricas de evaluación SVM (Stratify)
print('_'*75)
print("\n\n SVM (Stratify)\n")
print(f'Accuracy: {accuracy_svms:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_svms)
print('\nClassification Report:')
print(class_report_svms)

# Imprime las métricas de evaluación Decision Tree (Stratify)
print('_'*75)
print("\n\n Decision Tree (Stratify)\n")
print(f'Accuracy: {accuracy_dts:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_dts)
print('\nClassification Report:')
print(class_report_dts)

# Imprime las métricas de evaluación Logistic Regression (Stratify)
print('_'*75)
print("\n\n Logistic Regression (Stratify)\n")
print(f'Accuracy: {accuracy_lrs:.4f}')
print('\nConfusion Matrix:')
print(conf_matrix_lrs)
print('\nClassification Report:')
print(class_report_lrs)

#%% COMPARACIÓN

# Datos
modelos = ['Decision Tree', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
accuracy_holdout = [0.7448091091761554, 0.8526456798392499, 0.8519758874748827, 0.8526456798392499]
accuracy_stratify = [0.7488278633623576, 0.855324849296718, 0.8559946416610851, 0.854655056932351]

plt.figure(figsize=(10, 6))
plt.rcParams['axes.facecolor'] = '#228B22'  

# Puntos de holdout
plt.scatter(modelos, accuracy_holdout, color='red', marker='*', label='Holdout', s=500)

# Puntos de stratify 
plt.scatter(modelos, accuracy_stratify, color='#f5da2a', marker='*', label='Stratify', s=500)

# Configuraciones adicionales
plt.title('Accuracy de Modelos de Clasificación con Holdout y Stratify')
plt.xlabel('Modelo de Clasificación')
plt.ylabel('Accuracy')
plt.ylim(0.73, 0.87)
plt.legend(fontsize=14) 
plt.grid(True, linestyle='--', alpha=0.7)

plt.show()

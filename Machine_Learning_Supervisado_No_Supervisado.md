# Machine Learning: Supervisado vs No Supervisado

## MACHINE LEARNING SUPERVISADO
Aprende de datos etiquetados (X, Y). Predice Y dado X.

### A. CLASIFICACIÓN BINARIA
Y = 0 o 1, Sí/No

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Regresión Logística | P(Y=1) = 1/(1+e^(-X)); línea base probabilística | Predecir voto (prog/no prog) |
| K-Nearest Neighbors (KNN) | Clasifica como k vecinos más cercanos | Votantes similares votarán igual |
| Árbol de Decisión | Reglas if-then recursivas; interpretable | Comprender decisión voto |
| Random Forest | ~100 árboles; voto mayoritario | Robustez, importancia variables |
| Gradient Boosting (XGBoost) | Árboles secuenciales; minimiza residuos | Máxima precisión, ranking importancia |
| Support Vector Machine (SVM) | Hiperplano óptimo maximiza margen | Separación clara votos |
| Naive Bayes | Probabilidad condicional; asume independencia | Rápido, probabilidades calibradas |
| Neural Network | Capas no-lineales; aprende representaciones | Relaciones complejas |
| Ensemble Methods | Combina múltiples modelos (voting, stacking) | Robustez máxima |

### B. CLASIFICACIÓN MULTICLASE
Y = 3+ categorías

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Regresión Multinomial | Probabilidad multiclase; generalización logística | Predecir partido (Prog/Cons/Otro) |
| Árbol de Decisión (Multi) | Particiones recursivas múltiples clases | Reglas elección partido |
| Random Forest (Multi) | Árboles votan por clase | Clasificación multi-partido robusta |
| Gradient Boosting (Multi) | Boosting multiclase | Máxima precisión multi-partidos |
| SVM (One-vs-Rest) | 3+ SVM binarias combinadas | Separación multi-clase |
| Neural Network (Softmax) | Output softmax normaliza a probabilidades | Capas no-lineales multi-clase |
| Multinomial Naive Bayes | P(Clase|Features) multiclase | Rápido, múltiples partidos |

### C. CLASIFICACIÓN ORDINAL
Y ordinal: bajo < medio < alto

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Regresión Ordinal (Logit Ordinal) | Y ordinal; orden importa | Predecir aprobación (baja-med-alta) |
| Árbol de Decisión Ordinal | Respeta orden clases | Reglas jerarquizadas |

### D. REGRESIÓN
Predecir valor numérico continuo

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Regresión Lineal (OLS) | Y = a + bX; mínimos cuadrados | Relación voto-educación, baseline |
| Ridge Regression | Regularización L2; penaliza coefs grandes | Multicolinealidad |
| Lasso Regression | Regularización L1; algunos coefs = 0 | Selecciona variables automáticamente |
| Elastic Net | Combina Ridge + Lasso | Balance entre Ridge y Lasso |
| Polynomial Regression | Polinomios grado 2+ | Relaciones curvilíneas voto-edad |
| Spline Regression | Polinomios por segmento | Relación no-lineal suave |
| Kernel Ridge Regression | Ridge en espacio transformado (no-lineal) | Relaciones no-lineales complejas |
| SVR (Support Vector Regression) | SVM para regresión; margen epsilon | Predicción robusta % voto |
| KNN (Regresión) | Promedia k vecinos más cercanos | Votantes similares, promedio |
| Árbol de Decisión (Regresión) | Promedio por hoja terminal | Reglas para predecir % voto |
| Random Forest (Regresión) | Promedio ~100 árboles | Predicción robusta % voto |
| Gradient Boosting (Regresión) | Boosting secuencial para regresión | Máxima precisión predicción |
| Neural Network (Regresión) | Capas continuas; salida lineal | Relaciones no-lineales complejas |
| Gaussian Process (GP) | Distribución bayesiana; intervalos confianza | Predicción con incertidumbre |

### E. SERIES DE TIEMPO
Predecir valores secuenciales temporales

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| ARIMA | AutoRegressive Integrated Moving Average | Proyectar voto mensual |
| SARIMA | ARIMA + estacionalidad | Voto con ciclos electorales |
| Exponential Smoothing (Holt-Winters) | Peso reciente > pasado | Tendencia voto corto plazo |
| Prophet (Facebook) | Tendencia + estacionalidad; holidays | Voto con eventos políticos |
| LSTM (RNN) | Memoria largo plazo; secuencias | Tendencias complejas voto |
| Seasonal Decomposition | Separa Tendencia + Estacional + Residual | Entender componentes |
| Vector Autoregression (VAR) | Múltiples series simultáneamente | Voto + Desempleo + Aprobación |

### F. DETECCIÓN DE ANOMALÍAS (Con etiquetas)

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| One-Class SVM | Semi-supervisado; frontera alrededor mayoría | Fraude electoral potencial |
| Neural Network (Autoencoder Superviso) | Etiquetado; detecta anomalías conocidas | Patrones fraude identificados |

### G. MODELOS CAUSALES SUPERVISADO
Mezcla causal + machine learning

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Causal Forest | Árboles para estimar efectos heterogéneos | Efecto política varía por grupo |
| BART (Bayesian Additive Reg Trees) | Árboles aditivos bayesianos | Efectos heterogéneos causal |
| Double Machine Learning (DML) | Partialing out ML; controla endogeneidad | Estima efecto causal robusto |
| X-Learner, T-Learner | Metalearners para heterogeneidad | Efecto diferencial por grupo |

---

## MACHINE LEARNING NO SUPERVISADO
Aprende de datos SIN etiquetas. Descubre estructura.

### A. CLUSTERING / SEGMENTACIÓN
Agrupar observaciones similares

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| K-Means | k centros; minimiza distancia intra-cluster | 5 segmentos votantes: urban-joven, rural-viejo |
| Hierarchical Clustering | Árbol agrupamientos; dendrograma | Dendrograma distritos electorales |
| DBSCAN | Densidad espacial; clusters irregulares | Hotspots voto, grupos naturales |
| Gaussian Mixture Models (GMM) | Mezcla gaussiana probabilística | Probabilidad pertenencia cluster |
| Spectral Clustering | Similaridad no-lineal; grafo normalizado | Grupos complejos no separables |
| Agglomerative Clustering | Bottom-up; fusiona clusters similares | Agrupación jerárquica distritos |
| Affinity Propagation | Exemplars representativos | Representantes cada cluster |
| Self-Organizing Maps (SOM) | Mapa auto-organizado | Visualización 2D votantes |

### B. REDUCCIÓN DE DIMENSIONALIDAD
Comprimir variables; visualizar en 2D/3D

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Principal Component Analysis (PCA) | Combinación lineal; máxima varianza | Reduce 20 vars a 3 PCs |
| t-SNE | No-lineal preserva distancias; visualiza | Visualizar clusters en 2D |
| UMAP | No-lineal preserva topología | Mejor que t-SNE, mantiene estructura |
| Factor Analysis | Factores latentes subyacentes | 5 factores detrás preferencia voto |
| Independent Component Analysis (ICA) | Componentes independientes | Señales separadas, no correladas |
| Feature Selection (SelectKBest) | Top k variables más predictivas | Top 10 de 50 variables importantes |
| Recursive Feature Elimination (RFE) | Elimina iterativamente var menos importante | Selecciona variables paso a paso |

### C. ANÁLISIS DE ASOCIACIÓN
Descubrir reglas entre items

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Apriori | Reglas si-entonces frecuentes | Si vota prog, probable educación alta |
| Eclat | Búsqueda en profundidad itemsets | Combinaciones voto-características |
| FP-Growth | Crecimiento patrón frecuente | Patrones voto eficientemente |

### D. ANÁLISIS DE RED / GRAFOS
Estructura relaciones entre actores

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Análisis de Centralidad | Importancia nodos; grado, cercanía, intermediación | Políticos influyentes |
| Detección de Comunidades | Agrupa nodos conectados | Coaliciones políticas |
| Modularidad | Mide fuerza comunidades | Cohesión coalición |
| Link Prediction | Predice conexiones futuras | Próximas alianzas políticas |
| PageRank | Importancia en red iterativa | Ranking influencia política |
| HITS | Hubs y autoridades | Líderes vs amplificadores |

### E. TOPIC MODELING / TEXT MINING
Descubrir temas en corpus texto sin etiquetas

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Latent Dirichlet Allocation (LDA) | Distribución Dirichlet; temas latentes | 5 temas en discursos políticos |
| Non-Negative Matrix Factorization (NMF) | Factorización matriz; no-negativo | Temas más interpretables |
| Latent Semantic Analysis (LSA) | Descomposición SVD; semántica | Similitud conceptual |
| Top2Vec | Embeddings + clustering | Temas automáticos |
| BERTopic | BERT + clustering | Temas dinámicos |

### F. ANOMALY DETECTION (No Superviso)
Identificar outliers sin etiquetas

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Isolation Forest | Aislamiento recursivo | Fraude electoral potencial |
| Local Outlier Factor (LOF) | Densidad local | Baja densidad = anomalía |
| Elliptic Envelope (Robust Cov) | Covarianza robusta | Elipsoide robusto |
| Autoencoder (No Superviso) | Codif-decodif no-lineal | Error reconstrucción = anomalía |
| Z-Score, IQR, Mahalanobis | Métodos estadísticos | 3-sigma, rango intercuartil |
| DBSCAN | Densidad; puntos aislados = anomalía | Outliers densidad baja |

### G. PROYECCIONES / EMBEDDINGS
Representar objetos en espacio latente

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Word2Vec | Skip-gram/CBOW; semántica palabra | Similitud conceptual políticos |
| GloVe | Factorización matriz; co-ocurrencia | Embeddings palabra robusto |
| FastText | Subword embeddings; palabras raras | Trata OOV |
| Sentence Embeddings (USE, SBERT) | Oración completa; semántica | Similitud discursos políticos |
| Image Embeddings (CNN) | Redes convolucionales; imágenes | Clasificar imágenes política |
| Graph Embeddings (Node2Vec) | Paseos aleatorios; nodos grafo | Embeddings actores políticos |

### H. MODELADO PROBABILÍSTICO
Aprender distribuciones de datos

| Modelo | Definición Breve | Uso Político |
|--------|-----------------|-------------|
| Gaussian Mixture Models (GMM) | Mezcla gaussiana; probabilística | Segmentos votantes, probabilidades |
| Hidden Markov Models (HMM) | Estados ocultos; transiciones | Cambios voto latente |
| Bayesian Networks | Grafo probabilístico; dependencias | Estructura causal votantes |
| Variational Autoencoders (VAE) | Autoencoder bayesiano; generativo | Generar votantes sintéticos |
| Generative Adversarial Networks (GAN) | Generador vs discriminador | Datos electorales sintéticos |

### I. VALIDACIÓN NO SUPERVISO
Evaluar calidad clustering sin etiquetas

| Métrica | Definición Breve | Interpretar |
|---------|-----------------|-----------|
| Silhouette Score | Cohesión vs separación (-1 a 1) | >0.5 = clusters buenos |
| Davies-Bouldin Index | Ratio similitud intra vs inter cluster | Menor = mejor |
| Calinski-Harabasz Index | Ratio dispersión inter vs intra | Mayor = mejor |
| Inertia / Within-Cluster Sum of Squares | Distancia intra-cluster | Menor = mejor (elbow) |
| Gap Statistic | Compara con distribución aleatoria | Determina k óptimo |
| Dunn Index | Ratio mínima inter vs máxima intra | Mayor = mejor separación |

---

## TABLA COMPARATIVA: SUPERVISADO vs NO SUPERVISADO

| Aspecto | Supervisado | No Supervisado |
|--------|-------------|----------------|
| Datos | Etiquetados (X, Y) | Sin etiquetas (X) |
| Objetivo | Predecir Y dado X | Descubrir estructura en X |
| Ejemplos | Voto (Y) ~ educación (X) | Clusters votantes, temas discursos |
| Evaluación | Métricas: AUC, RMSE, Accuracy | Silhouette, Davies-Bouldin, Elbow |
| Riesgos | Overfitting, sesgo etiquetas | Clusters espurios, interpretación |
| Usos Político | Predicción, targeting | Segmentación, descubrimiento |
| Complejidad | Moderada | Alta (validar manualmente) |
| Ejemplos Principales | Logit, RF, XGBoost, LSTM, SVM | K-Means, DBSCAN, LDA, PCA, t-SNE |

---

## FLUJO TÍPICO ANÁLISIS POLÍTICO

### 1. EXPLORATORIO (No Superviso)
- PCA, t-SNE -> visualizar estructura
- K-Means, DBSCAN -> descubrir clusters votantes
- LDA -> temas discursos políticos

### 2. PREDICTIVO (Superviso)
- Logit, Random Forest -> predecir voto
- XGBoost -> máxima precisión
- SHAP -> explicar predicción

### 3. CAUSAL (Superviso Causal)
- Causal Forest -> efectos heterogéneos
- BART -> heterogeneidad por grupo
- DML -> estima efecto causal robusto

### 4. COMUNICACIÓN
- Visualizar clusters, importancia variables
- Reportar AUC, Silhouette, efectos

---

Creado para: Doctores en Ciencia Política y Especialistas en Políticas Públicas

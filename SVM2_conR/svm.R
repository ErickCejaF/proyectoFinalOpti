library(e1071)
library(data.table) ## or library(reshape2)
library(dplyr)
library(unikn)

songs <- data.frame(read.csv(file = '/Users/erickcejafuentes/DataspellProjects/Optimizacion convexa/ProyectoFinal/SVM2_conR/outputs/components_7/out.csv'))

y_label <- songs %>% select(label)

working_songs_pca <- songs %>% select(X1, X2)

data <- data.frame(working_songs_pca, y_label)

ggplot(data) +
  geom_point(aes(x = X1, y = X2, color = factor(label)), size = 2) +
  theme_bw()

data$label <- as.factor(data$label)

modelo_svm <- svm(formula = label ~ X1 + X2, data = data, kernel = "radial",
                  cost = 0.5,gamma= 1, scale = FALSE)

plot(modelo_svm, data)



# Para que la función svm() calcule el Support Vector Classifier,
# se tiene que indicar que la función kernel es lineal.
modelo_svm_rbf <- tune("svm", label ~ X1 + X2,
                       data = data, kernel = "radial",
                       ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 20, 400),
                                     gamma = c(0.5, 1, 2, 3, 4, 5, 10, 1000)))

summary(modelo_svm_rbf)

ggplot(data = modelo_svm_rbf$performances, aes(x = cost, y = error, color = as.factor(gamma))) +
  geom_line() +
  geom_point() +
  labs(title = "Error de clasificación vs hiperparámetros C y gamma", color = "gamma") +
  theme_bw() +
  theme(legend.position = "bottom")

# Se interpolar puntos dentro del rango de los dos predictores X1 y X2.
# Estos nuevos puntos se emplean para predecir la variable respuesta acorde
# al modelo y así colorear las regiones que separa el hiperplano.

# Rango de los predictores
rango_X1 <- range(data$X1)
rango_X2 <- range(data$X2)

# Interpolación de puntos
new_x1 <- seq(from = rango_X1[1], to = rango_X1[2], length = 75)
new_x2 <- seq(from = rango_X2[1], to = rango_X2[2], length = 75)
nuevos_puntos <- expand.grid(X1 = new_x1, X2 = new_x2)

# Predicción según el modelo de los nuevos puntos
predicciones <- predict(object = modelo_svm_rbf, newdata = nuevos_puntos)

# Se almacenan los puntos predichos para el color de las regiones en un dataframe
color_regiones <- data.frame(nuevos_puntos, y = predicciones)

ggplot() +
  # Representación de las 2 regiones empleando los puntos y coloreándolos
  # según la clase predicha por el modelo
  geom_point(data = color_regiones, aes(x = X1, y = X2, color = factor(y_label)),
             size = 0.5) +
  # Se añaden las observaciones
  geom_point(data = data, aes(x = X1, y = X2, color = factor(label)),
             size = 2.5) +
  # Se identifican aquellas observaciones que son vectores soporte
  geom_point(data = data[modelo_svm_rbf$index,],
             aes(x = X1, y = X2, color = as.factor(label)),
             shape = 21, colour = "black",
             size = 2.5) +
  theme_bw() +
  theme(legend.position = "none")


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

df <-  data.frame(read.csv(file = '/Users/erickcejafuentes/DataspellProjects/Optimizacion convexa/ProyectoFinal/SVM2_conR/music_data.csv'))
head(df)

library(caTools)
set.seed(1234)
split <- sample.split(df$label, SplitRatio = 0.80)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)

# Feature Scaling 1 age, 2 is salary
training_set[,1:13] <-  scale(training_set[,3:20])
test_set[,1:13] <-  scale(test_set[,3:20])

# install.packages("caret")
library(caret)
library(e1071)
pca <- preProcess(x = training_set[-14], method = "pca", pcaComp = 2)
training_set <- predict(pca, training_set)
# put customer segment in to the last postion or column
training_set <- training_set[c(2,3,1)]
library(tidyverse)
library(caTools)
library(caret)
library(ggplot2)
library(e1071)
library(dplyr)
library(nnet)

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# Main code

# working directory
setwd("/Users/erickcejafuentes/DataspellProjects/Optimizacion convexa/ProyectoFinal")

# reading the main file
df <- data.frame(read.csv(file = 'datasets/music_data.csv'))

# removing the second column
df <- select(df, -2)

########################################################################################################################

# Splitting the train and set
split <- sample.split(df$label, SplitRatio = 0.80)
training_set <- subset(df, split == TRUE)
test_set <- subset(df, split == FALSE)


training_set[, 2:29] <- scale(training_set[, 2:29])
test_set[, 2:29] <- scale(test_set[, 2:29])

pca <- preProcess(x = training_set[-1], method = "pca", pcaComp = 2)

########################################################################################################################

training_set_predicted <- predict(pca, training_set)
training_set_predicted <- training_set_predicted[c(2, 3, 1)]
training_set_predicted$label <- as.factor(training_set_predicted$label)

test_set_predicted <- predict(pca, test_set)
test_set_predicted <- test_set_predicted[c(2, 3, 1)]
test_set_predicted$label <- as.factor(test_set_predicted$label)

########################################################################################################################


multinom_model <- multinom(label ~ ., data = training_set_predicted)

multinom_pred <- predict(multinom_model, test_set_predicted)

confusion_matrix_tot <- confusion_matrix_info(multinom_pred, test_set_predicted)
summary(confusion_matrix)

graph_mesh(training_set_predicted, multinom_model, "Multinomial training set")

########################################################################################################################
# SVM classifier

classifier <- svm(formula = label ~ .,
                  data = training_set_predicted,
                  type = "C-classification",
                  kernel = "radial",
                  gamma = 10,
                  cost = 1)


cofwsfkgu <- confusion_matrix_info(predict(classifier, newdata = test_set_predicted[-3]), test_set_predicted)

graph_mesh(training_set_predicted, classifier, "SVM (Training Set)")

########################################################################################################################
#functions

graph_mesh <- function(data_to_graph, classiffier_method, title) {

  set <- data_to_graph

  X1 <- seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
  X2 <- seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)

  grid_set <- expand.grid(X1, X2)
  colnames(grid_set) <- c('PC1', 'PC2')
  genres <- predict(classiffier_method, newdata = grid_set)

  plot <- ggplot(set,
                 main = title,
                 xlab = 'PC1', ylab = 'PC2',
                 xlim = range(X1), ylim = range(X2))

  plot <- plot + geom_point(data = grid_set, aes(x = PC1, y = PC2, colour = genres))

  plot <- plot + geom_point(color = 'black', shape = 21, size = 2, data = set,
                            aes(x = PC1, y = PC2, fill = as.factor(label)), show.legend = F)

  plot
}

confusion_matrix_info <- function(data, test_set) {
  print(table(data, test_set$label))

  confusion_matrix <- confusionMatrix(data, test_set$label)

  print(paste(c(confusion_matrix$overall['Accuracy'] * 100, "% of accuracy"), collapse = " "))

  return(confusion_matrix)
}

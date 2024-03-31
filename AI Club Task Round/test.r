# Install necessary packages
install.packages(c("dplyr", "caret", "ROCR", "neuralnet"))

# Load libraries
library(dplyr)
library(caret)
library(ROCR)
library(neuralnet)

# Load Iris dataset
data(iris)

# Split data into train and test sets
set.seed(42)
train_index <- createDataPartition(iris$Species, p = 0.8, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

# Standardize features
standardize <- function(x) {
  (x - mean(x)) / sd(x)
}
train_data_scaled <- as.data.frame(lapply(train_data[, -5], standardize))
test_data_scaled <- as.data.frame(lapply(test_data[, -5], standardize))

# Train logistic regression model
logistic_model <- train(Species ~ ., data = train_data_scaled, method = "glm", trControl = trainControl(method = "cv"))

# Predictions
logistic_predictions <- predict(logistic_model, newdata = test_data_scaled)

# Evaluation metrics
accuracy <- confusionMatrix(logistic_predictions, test_data$Species)$overall["Accuracy"]
precision <- confusionMatrix(logistic_predictions, test_data$Species)$byClass["Pos Pred Value"]
recall <- confusionMatrix(logistic_predictions, test_data$Species)$byClass["Sensitivity"]
f1_score <- confusionMatrix(logistic_predictions, test_data$Species)$byClass["F1"]

# Save logistic regression model
saveRDS(logistic_model, "logistic_regression_model.rds")

# Load logistic regression model
logistic_regression_model <- readRDS("logistic_regression_model.rds")

# Train neural network
neural_network <- neuralnet(Species ~ ., data = train_data_scaled, hidden = c(100, 100, 100))

# Predictions
nn_predictions <- predict(neural_network, test_data_scaled)

# Evaluate neural network
nn_accuracy <- sum(nn_predictions$class == test_data$Species) / nrow(test_data)

# Model comparison plot
accuracy_comparison <- c(accuracy, nn_accuracy)
model_names <- c("Logistic Regression", "Neural Network")
barplot(accuracy_comparison, names.arg = model_names, ylim = c(0, 1), 
        main = "Model Comparison: Logistic Regression vs Neural Network", 
        ylab = "Accuracy", col = "skyblue", border = "black")

# Visuals
plot(test_data_scaled[, 1], test_data_scaled[, 2], col = logistic_predictions, 
     main = "Predicted Iris Species", xlab = "Sepal Length", ylab = "Sepal Width", pch = 19)
legend("topright", legend = levels(as.factor(iris$Species)), col = 1:3, pch = 19, bty = "n")

# Confusion matrix
conf_matrix <- table(logistic_predictions, test_data$Species)
heatmap(conf_matrix, Colv = NA, Rowv = NA, col = heat.colors(12), margins = c(5, 5), 
        main = "Confusion Matrix", xlab = "Predicted labels", ylab = "True labels")

# ROC curve
pred <- prediction(predict(logistic_model, newdata = test_data_scaled), test_data$Species)
perf <- performance(pred, "tpr", "fpr")
plot(perf, col = rainbow(3), main = "ROC Curve")
abline(0, 1, lty = 2)

# F1 Score
f1_score <- c(accuracy, precision, recall, f1_score)
metrics <- c("Accuracy", "Precision", "Recall", "F1 Score")
barplot(f1_score, names.arg = metrics, ylim = c(0, 1), main = "Evaluation Metrics", 
        ylab = "Score", col = "salmon", border = "black")

# Set the working directory
setwd("C:/Users/user/Downloads/Amili")

# Load necessary libraries
library(caret)
library(readxl)
library(randomForest)
library(dplyr)
library(ggplot2)
library(xgboost)
library(e1071)
library(reshape2)
library(skimr)
library(gridExtra)
library(grid)


# Load the datasets
taxonomy <- read_excel("taxonomy.xlsx")
age <- read.csv("age.csv")

# Rename the first column to 'ID' for both datasets
colnames(taxonomy)[1] <- "ID"
colnames(age)[1] <- "ID"

# Merge the two datasets by the sample IDs
df <- merge(taxonomy, age, by = "ID")

# Drop the 'ID' column using dplyr
df <- df %>% select(-ID)

# Check the shape (dimensions) of the dataset
cat("Dimensions of the dataset: ", dim(df))




### Feature Selection ###

# Define the features (microbiome profiles) and target variable (age)
X <- df[, -which(names(df) == "Age")]
y <- df$Age

# Remove near-zero variance predictors
nzv <- nearZeroVar(X)
X_filtered <- X[, -nzv]

# Check the dimensions after filtering
print(dim(X_filtered))

# Find highly correlated features (threshold can be adjusted)
correlation_matrix <- cor(X_filtered)
highly_correlated <- findCorrelation(correlation_matrix, cutoff = 0.9)

# Remove highly correlated features
X_filtered_no_corr <- X_filtered[, -highly_correlated]

# Check dimensions after removing correlated features
print(dim(X_filtered_no_corr))

# Scale the features
X_scaled <- scale(X_filtered_no_corr)

### Recursive Feature Elimination (RFE) ###

# Set up control parameters for RFE with Random Forest
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)

# Run RFE to select the top 10 features using Random Forest
set.seed(123)
rfe_results <- rfe(X_scaled, y, sizes = c(1:10), rfeControl = control)

# Print the results of the RFE
print(rfe_results)

# Plot the RFE results
plot(rfe_results, type = c("g", "o"))

# Feature by rank
top_10_features <- head(predictors(rfe_results), 10)

### Visualization of Top 10 Features ###

# Create a data frame for plotting
importance_df <- data.frame(
  Feature = top_10_features,
  Importance = seq(1, 10)
)

# Plot the top 10 features using ggplot2
ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "#BB9AB1") +
  coord_flip() +  
  labs(title = "Top 10 Selected Features by RFE", x = "Features", y = "Importance") +
  theme_minimal() +
  theme(
    axis.title.x = element_text(size = 14),  
    axis.title.y = element_text(size = 14),  
    axis.text.x = element_text(size = 14),   
    axis.text.y = element_text(size = 14)    
  )



### Exploration For Top 10 Selected Features ###

# Extract the 10 most important features from the original dataset
df_top_10 <- df[, c(top_10_features, "Age")]

# View the first few rows 
cat("First 6 rows of the dataset:\n")
head(df_top_10)

# Get the data types of each column
cat("Data types of each column:\n")
print(sapply(df_top_10, class))

# Get basic statistics (summary) for numerical columns
cat("Summary statistics for numerical columns:\n")
skim(df_top_10)

# View the first 6 rows
head(df_top_10)

# Check for missing values
cat("Number of missing values in each column:\n")
colSums(is.na(df_top_10))

# Check for duplicated rows
cat("Number of duplicated rows: ", sum(duplicated(df_top_10)))

# List of numeric columns
num_cols <- names(df_top_10)[sapply(df_top_10, is.numeric)]

# Create an empty list to store the plots
plot_list <- list()

# Loop through each column and create a histogram with KDE
for (col in num_cols) {
  # Create the histogram with KDE using ggplot2
  p <- ggplot(df_top_10, aes(x = .data[[col]])) +
    geom_histogram(aes(y = ..density..), binwidth = 10, fill = "#BB9AB1", color = "black", alpha = 0.7) + 
    geom_density(color = "darkblue", linewidth = 1) + 
    labs(x = col, y = "Density") +  
    theme_minimal() +
    theme(
      plot.title = element_blank(), 
      axis.text.x = element_text(angle = 45, hjust = 1),  
      axis.text.y = element_text(size = 8),  
      plot.margin = unit(c(5, 5, 5, 5), "mm")  
    )
  
  # Add each plot to the list
  plot_list[[col]] <- p
}

# Set the number of columns and rows for better readability
n_col <- 3 
n_row <- ceiling(length(plot_list) / n_col)  

# Arrange the plots into a grid
grid.arrange(
  do.call("arrangeGrob", c(plot_list, ncol = n_col, nrow = n_row)),
  top = textGrob("Distribution of Gut Microbiome Profiles", gp = gpar(fontsize = 16, fontface = "bold"))  # Main title for all plots
)

### Outliers Handling ###

# Visualize outliers using boxplots for numerical variables
num_cols <- sapply(df_top_10, is.numeric)
df_numeric <- df_top_10[, num_cols]
df_melt <- melt(df_numeric)

ggplot(df_melt, aes(x = variable, y = value)) +
  geom_boxplot() +
  labs(title = "Boxplot for Outlier Detection", x = "Feature", y = "Value") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Separate the target variable 'Age' from the numeric features
df_features_only <- df_numeric[, -which(names(df_numeric) == "Age")]

# Log-transform the numeric features (handling zeros with log1p) excluding 'Age'
df_log_transformed <- log1p(df_features_only)

# Melt the log-transformed data for plotting
df_melt_log <- melt(df_log_transformed)

# Create boxplot for log-transformed data (excluding 'Age' from log transformation)
ggplot(df_melt_log, aes(x = variable, y = value)) +
  geom_boxplot() +
  labs(title = "Boxplot for Outlier Detection (Log-Transformed Data)", x = "Feature", y = "Log-Transformed Value") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Add 'Age' back to the log-transformed dataframe
df_log_transformed <- cbind(df_log_transformed, Age = df_numeric$Age)



### Model Development ###

# Split the data into training (70%) and test sets (30%)
set.seed(123)
train_index <- createDataPartition(df_log_transformed$Age, p = 0.7, list = FALSE)
train_data <- df_log_transformed[train_index, ]
test_data <- df_log_transformed[-train_index, ]

# Define the features (X) and target variable (y) for both training and test sets
X_train <- train_data[, -which(names(train_data) == "Age")]
y_train <- train_data$Age
X_test <- test_data[, -which(names(test_data) == "Age")]
y_test <- test_data$Age

# Preprocess the data to scale X_train and X_test
preprocess_params <- preProcess(X_train, method = c("center", "scale"))

# Apply the scaling to X_train and X_test
X_train <- predict(preprocess_params, X_train)
X_test <- predict(preprocess_params, X_test)


### Random Forest Model Training ###

# Train the Random Forest model using the training data
set.seed(123)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500, importance = TRUE)

# Print the model summary
print(rf_model)

# Evaluate the model using the test data
y_pred <- predict(rf_model, X_test)

# Calculate evaluation metrics for Random Forest
rmse_rf <- sqrt(mean((y_pred - y_test)^2))
mae_rf <- mean(abs(y_pred - y_test))
r_squared_rf <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

# Print Random Forest evaluation results
cat("Random Forest RMSE:", rmse_rf, "\n")
cat("Random Forest MAE:", mae_rf, "\n")
cat("Random Forest R-squared:", r_squared_rf, "\n")

# Set up the grid for hyperparameter tuning
rf_grid <- expand.grid(
  mtry = c(2, 3, 4, 5)  
)

# Set up control for cross-validation
control <- trainControl(
  method = "cv",  
  number = 5,     
  verboseIter = TRUE  
)

# Train the Random Forest model with hyperparameter tuning
set.seed(123)
rf_tune <- train(
  x = X_train,
  y = y_train,
  method = "rf",  
  trControl = control,
  tuneGrid = rf_grid,
  ntree = 500,  
  metric = "RMSE"  
)

# Print the best hyperparameters
print(rf_tune$bestTune)

### Evaluate the Tuned Random Forest Model ###

# Predict using the tuned model on the test set
y_pred_rf_tuned <- predict(rf_tune, newdata = X_test)

# Calculate evaluation metrics for the tuned Random Forest model
rmse_rf_tuned <- sqrt(mean((y_pred_rf_tuned - y_test)^2))
mae_rf_tuned <- mean(abs(y_pred_rf_tuned - y_test))
r_squared_rf_tuned <- 1 - sum((y_test - y_pred_rf_tuned)^2) / sum((y_test - mean(y_test))^2)

# Print evaluation results for the tuned Random Forest model
cat("Tuned Random Forest RMSE:", rmse_rf_tuned, "\n")
cat("Tuned Random Forest MAE:", mae_rf_tuned, "\n")
cat("Tuned Random Forest R-squared:", r_squared_rf_tuned, "\n")


### XGBoost Model Training ###

# Convert the features and target to matrix format for XGBoost
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)
y_train_matrix <- as.numeric(y_train)
y_test_matrix <- as.numeric(y_test)

# Set up the parameters for XGBoost
params <- list(
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the XGBoost model
set.seed(123)
xgb_model <- xgboost(
  data = X_train_matrix,
  label = y_train_matrix,
  params = params,
  nrounds = 500,
  verbose = 0
)

# Evaluate the XGBoost model using the test data
y_pred_xgb <- predict(xgb_model, X_test_matrix)

# Calculate evaluation metrics for XGBoost
rmse_xgb <- sqrt(mean((y_pred_xgb - y_test_matrix)^2))
mae_xgb <- mean(abs(y_pred_xgb - y_test_matrix))
r_squared_xgb <- 1 - sum((y_test_matrix - y_pred_xgb)^2) / sum((y_test_matrix - mean(y_test_matrix))^2)

# Print XGBoost evaluation results
cat("XGBoost RMSE:", rmse_xgb, "\n")
cat("XGBoost MAE:", mae_xgb, "\n")
cat("XGBoost R-squared:", r_squared_xgb, "\n")

### Hyperparameter Tuning with caret for XGBoost ###

# Define the grid of hyperparameters to search
xgb_grid <- expand.grid(
  nrounds = c(100, 200, 300),
  eta = c(0.01, 0.1, 0.3),
  max_depth = c(3, 6, 9),
  colsample_bytree = c(0.5, 0.7, 0.9),
  subsample = c(0.5, 0.7, 0.9),
  gamma = 0,
  min_child_weight = 1
)

# Set up control for cross-validation
control <- trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE
)

# Train the XGBoost model with hyperparameter tuning
set.seed(123)
xgb_tune <- train(
  x = X_train_matrix,
  y = y_train_matrix,
  method = "xgbTree",
  trControl = control,
  tuneGrid = xgb_grid,
  metric = "RMSE"
)

# Print the best hyperparameters
print(xgb_tune$bestTune)

### Evaluate the tuned XGBoost model ###

# Predict using the best tuned XGBoost model
y_pred_xgb_tuned <- predict(xgb_tune, newdata = X_test_matrix)

# Calculate evaluation metrics for tuned XGBoost
rmse_xgb_tuned <- sqrt(mean((y_pred_xgb_tuned - y_test_matrix)^2))
mae_xgb_tuned <- mean(abs(y_pred_xgb_tuned - y_test_matrix))
r_squared_xgb_tuned <- 1 - sum((y_test_matrix - y_pred_xgb_tuned)^2) / sum((y_test_matrix - mean(y_test_matrix))^2)

# Print evaluation results for the tuned XGBoost model
cat("Tuned XGBoost RMSE:", rmse_xgb_tuned, "\n")
cat("Tuned XGBoost MAE:", mae_xgb_tuned, "\n")
cat("Tuned XGBoost R-squared:", r_squared_xgb_tuned, "\n")



### SVR Model Training ###

# Train the SVR model using the training data
set.seed(123)
svr_model <- svm(x = X_train, y = y_train, type = "eps-regression", kernel = "radial")

# Print the model summary
print(summary(svr_model))

### Evaluate the SVR Model ###

# Predict the target variable on the test set
y_pred_svr <- predict(svr_model, X_test)

# Calculate evaluation metrics for SVR
rmse_svr <- sqrt(mean((y_pred_svr - y_test)^2))  
mae_svr <- mean(abs(y_pred_svr - y_test)) 
r_squared_svr <- 1 - sum((y_test - y_pred_svr)^2) / sum((y_test - mean(y_test))^2)  

# Print evaluation results for SVR
cat("SVR RMSE:", rmse_svr, "\n")
cat("SVR MAE:", mae_svr, "\n")
cat("SVR R-squared:", r_squared_svr, "\n")

### Hyperparameter Tuning with caret ###

# Set up the grid for hyperparameter tuning (sigma and C only)
svr_grid <- expand.grid(
  C = c(0.1, 1, 10),       # Regularization parameter
  sigma = c(0.01, 0.1, 1)  # Kernel coefficient for RBF (1/(2*gamma))
)

# Set up control for cross-validation
control <- trainControl(
  method = "cv",  
  number = 5,     
  verboseIter = TRUE  
)

# Train the SVR model with hyperparameter tuning
set.seed(123)
svr_tune <- train(
  x = X_train,
  y = y_train,
  method = "svmRadial",  
  trControl = control,   
  tuneGrid = svr_grid,   
  metric = "RMSE"        
)

# Print the best hyperparameters found during tuning
print(svr_tune$bestTune)

### Evaluate the Tuned SVR Model ###

# Predict using the tuned SVR model
y_pred_svr_tuned <- predict(svr_tune, newdata = X_test)

# Calculate evaluation metrics for the tuned SVR model
rmse_svr_tuned <- sqrt(mean((y_pred_svr_tuned - y_test)^2))  
mae_svr_tuned <- mean(abs(y_pred_svr_tuned - y_test))  
r_squared_svr_tuned <- 1 - sum((y_test - y_pred_svr_tuned)^2) / sum((y_test - mean(y_test))^2)  

# Print evaluation results for the tuned SVR model
cat("Tuned SVR RMSE:", rmse_svr_tuned, "\n")
cat("Tuned SVR MAE:", mae_svr_tuned, "\n")
cat("Tuned SVR R-squared:", r_squared_svr_tuned, "\n")


### Residual Plots ###
# Create residuals for each model
residuals_rf <- y_test - y_pred_rf_tuned  
residuals_xgb <- y_test - y_pred_xgb      
residuals_svr <- y_test - y_pred_svr    

# Create a combined data frame for residuals
residuals_df <- data.frame(
  Model = rep(c("Random Forest", "XGBoost", "SVR"), each = length(y_test)),
  Predicted = c(y_pred_rf_tuned, y_pred_xgb, y_pred_svr),
  Residuals = c(residuals_rf, residuals_xgb, residuals_svr)
)

# Plot residuals for each model
ggplot(residuals_df, aes(x = Predicted, y = Residuals, color = Model)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +  # Line at residual = 0
  facet_wrap(~ Model) +  
  labs(title = "Residual Plots for Each Model", x = "Predicted Values", y = "Residuals") +
  theme_minimal() +
  theme(legend.position = "none") 



### Actual vs Predicted Plot ###

# Random Forest: Actual vs. Predicted Plot
ggplot(data.frame(Actual = y_test, Predicted = y_pred_rf_tuned), aes(x = Actual, y = Predicted)) +
  geom_point(color = "#987D9A") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted - Random Forest", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# SVR: Actual vs. Predicted Plot
ggplot(data.frame(Actual = y_test, Predicted = y_pred_svr_tuned), aes(x = Actual, y = Predicted)) +
  geom_point(color = "#9BC4E2") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted - SVR", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()

# XGBoost: Actual vs. Predicted Plot
ggplot(data.frame(Actual = y_test, Predicted = y_pred_xgb_tuned), aes(x = Actual, y = Predicted)) +
  geom_point(color = "#FFEEAD") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Actual vs. Predicted - XGBoost", x = "Actual Values", y = "Predicted Values") +
  theme_minimal()


### Model Comparison ###


# Create a data frame to hold the evaluation metrics for all models
model_results <- data.frame(
  Model = c("Random Forest", "XGBoost", "SVR"),
  RMSE = c(rmse_rf, rmse_xgb_tuned, rmse_svr),
  MAE = c(mae_rf, mae_xgb_tuned, mae_svr),
  R_squared = c(r_squared_rf, r_squared_xgb_tuned, r_squared_svr)
)

# Melt the data frame to get it in long format for ggplot
model_results_long <- melt(model_results, id.vars = "Model", variable.name = "Metric", value.name = "Value")

# Plot for RMSE with values on top of bars
ggplot(subset(model_results_long, Metric == "RMSE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 4)), vjust = -0.5, color = "black", size = 4) +  # Display value on top
  labs(title = "RMSE Comparison by Model", x = "Model") +  # Remove y-axis title
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.ticks.y = element_blank(),  
        axis.text.y = element_blank(),  
        axis.title.y = element_blank()) + 
  scale_fill_manual(values = c("Random Forest" = "#FFEEAD", "XGBoost" = "#987D9A", "SVR" = "#9BC4E2")) +
  theme(legend.position = "none")  # Remove legend

# Plot for MAE with values on top of bars
ggplot(subset(model_results_long, Metric == "MAE"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 4)), vjust = -0.5, color = "black", size = 4) + 
  labs(title = "MAE Comparison by Model", x = "Model") +  
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.ticks.y = element_blank(),  
        axis.text.y = element_blank(),   
        axis.title.y = element_blank()) + 
  scale_fill_manual(values = c("Random Forest" = "#FFEEAD", "XGBoost" = "#987D9A", "SVR" = "#9BC4E2")) +
  theme(legend.position = "none") 

# Plot for R-squared with values on top of bars
ggplot(subset(model_results_long, Metric == "R_squared"), aes(x = Model, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(Value, 4)), vjust = -0.5, color = "black", size = 4) +  
  labs(title = "R-squared Comparison by Model", x = "Model") +  
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.ticks.y = element_blank(),  
        axis.text.y = element_blank(),   
        axis.title.y = element_blank()) + 
  scale_fill_manual(values = c("Random Forest" = "#FFEEAD", "XGBoost" = "#987D9A", "SVR" = "#9BC4E2")) +
  theme(legend.position = "none")  



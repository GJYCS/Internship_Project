# Library
library(tidyverse)
library(ggplot2)
library(caret)
library(glmnet)
library(dplyr)
library(broom)
library(car)
library(randomForest)
library(gbm)
library(MatchIt)
library(AER)
library(reshape2)
library(vcd)
library(FactoMineR)
library(pdp)

# Load the cleaned data
data <- read.csv("occupied_puf_21_cleaned.csv")
str(data)
# Now, the data includes 7029 obs. of 70 variables.

# Convert all categorical variables to factors
data$TENURE <- as.factor(data$TENURE)
data$HH62PLUS <- as.factor(data$HH62PLUS)
data$HHUNDER18 <- as.factor(data$HHUNDER18)
data$HHUNDER6 <- as.factor(data$HHUNDER6)
data$HHDEAR <- as.factor(data$HHDEAR)
data$HHDEYE <- as.factor(data$HHDEYE)
data$HHDREM <- as.factor(data$HHDREM)
data$HHDPHY <- as.factor(data$HHDPHY)
data$HHDDRS <- as.factor(data$HHDDRS)
data$HHDOUT <- as.factor(data$HHDOUT)
data$HHDONEPLUS <- as.factor(data$HHDONEPLUS)
data$NOHEAT <- as.factor(data$NOHEAT)
data$NOHOTWATER <- as.factor(data$NOHOTWATER)
data$ADDHEAT <- as.factor(data$ADDHEAT)
data$LEAKS <- as.factor(data$LEAKS)
data$MOLD <- as.factor(data$MOLD)
data$MUSTY <- as.factor(data$MUSTY)
data$RODENTS_UNIT <- as.factor(data$RODENTS_UNIT)
data$TOILET_BROK <- as.factor(data$TOILET_BROK)
data$ROACHES_NUM <- as.factor(data$ROACHES_NUM)
data$WALLHOLES <- as.factor(data$WALLHOLES)
data$FLOORHOLES <- as.factor(data$FLOORHOLES)
data$PEELPAINT <- as.factor(data$PEELPAINT)
data$ANIMS <- as.factor(data$ANIMS)
data$PA_FOOD <- as.factor(data$PA_FOOD)
data$PA_CASH <- as.factor(data$PA_CASH)
data$PA_OTHER <- as.factor(data$PA_OTHER)
data$PA_ANY <- as.factor(data$PA_ANY)
data$FOODINSECURE <- as.factor(data$FOODINSECURE)
data$UTIL_ELECTRIC <- as.factor(data$UTIL_ELECTRIC)
data$UTIL_GAS <- as.factor(data$UTIL_GAS)
data$UTIL_HEAT <- as.factor(data$UTIL_HEAT)
data$UTIL_WATER <- as.factor(data$UTIL_WATER)
data$UTIL_INCLUDED <- as.factor(data$UTIL_INCLUDED)
data$UTIL_NONE <- as.factor(data$UTIL_NONE)
data$INTERUPT_UTIL <- as.factor(data$INTERUPT_UTIL)
data$INTERUPT_PHONE <- as.factor(data$INTERUPT_PHONE)
data$INTERUPT_CELL <- as.factor(data$INTERUPT_CELL)
data$INTERUPT_NONE <- as.factor(data$INTERUPT_NONE)
data$EMERG400_RATE <- as.factor(data$EMERG400_RATE)
data$LEASENOW <- as.factor(data$LEASENOW)
data$RENTASSIST <- as.factor(data$RENTASSIST)
data$RENTASSIST_VOUCHER <- as.factor(data$RENTASSIST_VOUCHER)
data$RENTOUTSIDE <- as.factor(data$RENTOUTSIDE)
data$RENTFEES <- as.factor(data$RENTFEES)
data$RENTPAID <- as.factor(data$RENTPAID)
data$MISSRENT <- as.factor(data$MISSRENT)
data$ALTRENT_CREDIT <- as.factor(data$ALTRENT_CREDIT)
data$ALTRENT_SAVINGS <- as.factor(data$ALTRENT_SAVINGS)
data$ALTRENT_LOAN <- as.factor(data$ALTRENT_LOAN)
data$ALTRENT_ASSET <- as.factor(data$ALTRENT_ASSET)
data$ALTRENT_ONEPLUS <- as.factor(data$ALTRENT_ONEPLUS)
data$RENTBURDEN_CAT <- as.factor(data$RENTBURDEN_CAT)
data$CROWD_BDRM <- as.factor(data$CROWD_BDRM)
data$CROWD_RM <- as.factor(data$CROWD_RM)
data$HHCOVIDDIAG <- as.factor(data$HHCOVIDDIAG)

# Check data again
str(data)
# Standardizing Numeric Variables
numeric_col <- sapply(data, is.numeric)
data[numeric_col] <- scale(data[numeric_col])


# ---------------------
# Variable Selection via Lasso Regression
# ---------------------
set.seed(123)
# y is MUTIL
# Since is generated based on the following variables in the 2021 NYCHVS PUF dataset(s): 
# UTILCOSTS_SUMMER, UTILCOSTS_WINTER, UTILCOSTS_HEAT, and UTILCOSTS_WATER
# It is better to remove these variables
# Remove GRENT because is it the combined of RENT_AMOUNT and MUTIL
# UTILCOSTS_HEAT, UTILCOSTS_WATER already removed in data-cleaning process
# Also, remove all utility cost related variables
data <- data %>% 
  select(-UTILCOSTS_SUMMER, -UTILCOSTS_WINTER, -starts_with("UTIL_"), -GRENT)

all_covariates <- names(data)[names(data) != "MUTIL"]
x <- data.matrix(data[, all_covariates])
y <- data$MUTIL

# Cross-validation using k = 10-fold CV
cv_lasso <- cv.glmnet(x, y, alpha = 1, family = "gaussian")
best_lambda <- cv_lasso$lambda.min

# Fit the lasso model using best_lambda
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
# Extract selected covariates
included_covariates <- rownames(coef(best_model))[which(coef(best_model) != 0)]
included_covariates <- included_covariates[included_covariates != "(Intercept)"]

# Numeric covariates correlation check
numeric_covariates <- included_covariates[sapply(data[included_covariates], is.numeric)]
cor_matrix <- cor(data[, numeric_covariates], use = "pairwise.complete.obs")
cor_matrix
# There are no highly-correlated numeric variables.

# Categorical covariates correlation check using chi-squared
categorical_covariates <- included_covariates[sapply(data[included_covariates], is.factor)]

# chi-square
chi_sq <- function(var1, var2, data) {
  tbl <- table(data[[var1]], data[[var2]])
  test_result <- tryCatch({
    test <- chisq.test(tbl)
    return(test$p.value)})
}

# p-value
p_values <- matrix(nrow = length(categorical_covariates), 
                  ncol = length(categorical_covariates),
                  dimnames = list(categorical_covariates, categorical_covariates))
for (i in 1:length(categorical_covariates)) {
  for (j in 1:length(categorical_covariates)) {
    p_values[i, j] <- chi_sq(categorical_covariates[i], 
                             categorical_covariates[j], 
                             data)
  }
}
print(p_values)

# Exclude variable with p-value > 0.05
# 'HHUNDER6', 'HHDEAR', 'INTERUPT_NONE'
# Other excluded variables:
# 'CROWD_BDRM' because it is entirely dependent on 'CROWD_RM'.
included_covariates <- included_covariates[!included_covariates %in% 
                                             c("HHUNDER6",
                                               "HHDEAR",
                                               "INTERUPT_UTIL",
                                               "CROWD_BDRM")]

# I will keep the rest 11 variables and pass them to the next section!
print(included_covariates)

# Reorganize all included coveriates
included_covariates <- c(included_covariates, "MUTIL")
included_covariates
new_data <- data[, included_covariates]
# I will use this scaled data 'new_data' for all models to simplify my data preprocessing pipeline.
str(new_data)


# ---------------------
# General Linear Models (GLM)
# ---------------------
glm_model <- glm(MUTIL ~ ., data = new_data, family = gaussian())
# Result
summary(glm_model)
# Check for multicollinearity using VIF
vif(glm_model) # Any greater than 10? If not, passed


# ---------------------
# Advanced Machine Learning Models
# ---------------------
# Random Forests
rf_model <- randomForest(MUTIL ~ ., data = new_data, ntree = 100)
print(rf_model)

# Calculate importance for Variable importance plot use
importance_data <- importance(rf_model)


# Gradient Boosting Machines
gbm_model <- gbm(MUTIL ~ ., data = new_data, distribution = "gaussian", n.trees = 1000, 
                 interaction.depth = 3, shrinkage = 0.01, cv.folds = 5)
summary(gbm_model)

# Calculate predictions and RMSE for each model
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}
# Prediction
predictions_glm <- predict(glm_model, newdata = new_data, type = "response")
predictions_rf <- predict(rf_model, newdata = new_data)
predictions_gbm <- predict(gbm_model, newdata = new_data, n.trees = 1000, type = "response")
# RMSE
rmse_glm <- rmse(new_data$MUTIL, predictions_glm)
rmse_rf <- rmse(new_data$MUTIL, predictions_rf)
rmse_gbm <- rmse(new_data$MUTIL, predictions_gbm)
cat("RMSE for GLM:", rmse_glm, "\n")
cat("RMSE for RF:", rmse_rf, "\n")
cat("RMSE for GBM:", rmse_gbm, "\n")

# RF has the lowest RMSE. 

# ---------------------
# Propensity Score Matching (PSM)
# ---------------------
# TENURE: 1. Renter 2. Owner
new_data$TENURE <- factor(new_data$TENURE, levels = c(1, 2), labels = c("Renter", "Owner"))

# Modeling
matchit_model <- matchit(TENURE ~ ., data = new_data, method = "nearest")
summary(matchit_model, standardize = TRUE)

# Find all matched data
matched_data <- match.data(matchit_model)
# TENURE effect on MUTIL using glm
glm_matched <- glm(MUTIL ~ TENURE, data = matched_data, family = gaussian())
summary(glm_matched)


# ---------------------
# Visualization
# ---------------------

# 1. Variable Importance Plot for RF
varImpPlot(rf_model)

# 2. Relative influence Plot for GBM
par(las = 2)
par(cex.axis = 0.4)
gbm_importance <- summary(gbm_model)
par(cex.axis = 1)

# 3. Heatmap: I am interested in the correlations between MUTIL and all numerical predictors.
numeric_col <- sapply(new_data, is.numeric)
heatmap_data <- new_data[, numeric_col]
cor_matrix <- cor(heatmap_data, use = "pairwise.complete.obs", method = "pearson")
# ggplot
melted_cor_matrix <- melt(cor_matrix)
ggplot(melted_cor_matrix, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Heatmap", x = "", y = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), axis.text.y = element_text(angle = 45, vjust = 1))
  
# 4. Mosaic Plot: To visualize the relationship between TENURE and HHSIZE.
mosaic_plot <- mosaic(~ TENURE + HHSIZE, data = new_data,
                      main = "Mosaic Plot of Tenure by Household Size",
                      xlab = "Tenure", ylab = "Household Size")

# 5. Box Plots for Utility Costs by Tenure
ggplot(new_data, aes(x = TENURE, y = MUTIL, fill = TENURE)) +
  geom_boxplot() +
  labs(title = "Utility Costs by Tenure", x = "Tenure", y = "Monthly Utility Costs (MUTIL)") +
  theme_minimal()

# 6. PCA Plot: To reduce dimensionality and visualize the most significant variables affecting MUTIL.
pca_results <- PCA(new_data[, numeric_col], graph = FALSE)
plot(pca_results, choix = "ind", title = "PCA of Key Predictors")

# 7. Bar Chart: To compare average utility costs by tenure distribution.
ggplot(new_data, aes(x = TENURE, y = MUTIL, fill = TENURE)) +
  geom_bar(stat = "summary", fun = "mean") +
  labs(title = "Average Utility Costs by Tenure", x = "Tenure", y = "Average Monthly Utility Costs") +
  theme_minimal()

# 8. Interaction Effects Plot: Visualize how the interaction between TENURE and HHSIZE, affects MUTIL.
ggplot(new_data, aes(x = HHSIZE, y = MUTIL, color = TENURE)) +
  geom_point() +
  geom_smooth(method = "lm") +
  facet_wrap(~TENURE) +
  labs(title = "Interaction Effect of Household Size and Tenure on Utility Costs",
       x = "Household Size", y = "Monthly Utility Costs") +
  theme_minimal()

# 9. Distribution Plot: Utility Costs Across the presence of older adults
ggplot(new_data, aes(x = MUTIL, fill = HH62PLUS)) +
  geom_histogram(binwidth = 50, alpha = 0.6) +
  facet_wrap(~HH62PLUS) +
  labs(title = "Distribution of Utility Costs by Presence of Older Adults",
       x = "Monthly Utility Costs", y = "Frequency") +
  theme_minimal()



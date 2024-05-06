# Library
library(tidyverse)
library(ggplot2)
library(caret)
library(glmnet)
library(dplyr)

# Load the data
data <- read.csv("occupied_puf_21.csv")

# Remove column 1: Control number & all 'FW' & all allocation flag
data <- data %>% select(-CONTROL, -starts_with("FW"), -starts_with("I_"), -starts_with("FLG_"))
data

min(data)
# -3 to -1 is either not applicable, not reported, or not responded, I decide to replace them with NA
# Replace -3, -2, -1 with NA across all numeric columns
data <- data %>%
  mutate(across(where(is.numeric), ~if_else(.x < 0, NA_real_, .x)))

# Check for any blanks or NAs
check_missing_data <- function(data) {
  missing_data <- sapply(data, function(x) {
    na_count <- sum(is.na(x))
    data_summary <- c(num_missing = na_count,
                      percent_missing = na_count / nrow(data) * 100)
    return(data_summary)
  })
  missing_data <- as.data.frame(t(missing_data))
  missing_data <- rownames_to_column(missing_data, var = "variable")
  return(missing_data)
}

missing_data_summary <- check_missing_data(data)
print(missing_data_summary)

# I will drop the variables with at least 50% data missing (50% data available)
# I believe variables with a lot of missing values bring high uncertainty, so I would better to remove it instead of conducting basic imputation.
keep_variable <- sapply(data, function(x) {
  mean(is.na(x)) < 0.5
})
data <- data[, keep_variable]

# Then, I will conduct a median imputation for all missing values
data <- data %>%
  mutate(across(where(is.numeric), ~if_else(is.na(.x), median(.x, na.rm = TRUE), .x)))

# Recheck missing values
recheck_missing_data <- function(data) {
  missing_data <- sapply(data, function(x) {
    na_count <- sum(is.na(x))
    data_summary <- c(num_missing = na_count,
                      percent_missing = na_count / nrow(data) * 100)
    return(data_summary)
  })
  missing_data <- as.data.frame(t(missing_data))
  missing_data <- rownames_to_column(missing_data, var = "variable")
  return(missing_data)
}

# Re-check missing data summary
rechecked_missing_data_summary <- recheck_missing_data(data)
print(rechecked_missing_data_summary)


# In addition, I plan to use a hybrid method to check for extreme abnormal values / outliers
# First, use IQR to check for any outliers
check_outliers <- function(data) {
  outlier_summary <- sapply(data, function(x) {
    if(is.numeric(x)) {
      Q1 <- quantile(x, 0.25, na.rm = TRUE)
      Q3 <- quantile(x, 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      lower_bound <- Q1 - 1.5 * IQR
      upper_bound <- Q3 + 1.5 * IQR
      num_outliers <- sum(x < lower_bound | x > upper_bound, na.rm = TRUE)
      percent_outliers <- 100 * num_outliers / sum(!is.na(x))
    } else {
      num_outliers <- NA
      percent_outliers <- NA
    }
    return(c(num_outliers, percent_outliers))
  }, simplify = FALSE)
  outlier_summary_df <- tibble(
    variable = names(outlier_summary),
    num_outliers = sapply(outlier_summary, function(x) x[1]),
    percent_outliers = sapply(outlier_summary, function(x) x[2])
  )
  return(outlier_summary_df)
}

outlier_summary <- check_outliers(data)
print(outlier_summary)

# For convenience, I will only look into any variables with percent_outlier > 25%
significant_outliers <- outlier_summary %>% filter(percent_outliers > 25)
print(significant_outliers)

# Check all 3 significant outliers to see if all data is in the range or there any typo

# UTILCOSTS_SUMMER
# Topcode: 970, mean above topcode $1372
# Apply scaling to values over $970
count_above_970 <- sum(data$UTILCOSTS_SUMMER > 970, na.rm = TRUE)
count_above_970
# Only 38 entries, I will just remove them
data <- data[data$UTILCOSTS_SUMMER <= 970, ]
# Verify changes
max(data$UTILCOSTS_SUMMER)

# UTILCOSTS_WINTER
# Topcode: 1280, mean above topcode: 2207
# Apply scaling to values over $1280
count_above_1280 <- sum(data$UTILCOSTS_WINTER > 1280, na.rm = TRUE)
count_above_1280
# Only 22 entries, I will just remove them
data <- data[data$UTILCOSTS_WINTER <= 1280, ]
# Verify changes
max(data$UTILCOSTS_WINTER)

# MUTIL
min(data$MUTIL)
max(data$MUTIL)
# Should be between 0 and 9998. Passed

# All set. Save the cleaned data
write.csv(data, "occupied_puf_21_cleaned.csv", row.names = FALSE)


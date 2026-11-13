# Body Fat Prediction using R
# Baseline model: predict BodyFat from Age using simple linear regression.
# Run this script from the project root folder.

# 1. Install/load required package
if (!requireNamespace("ggplot2", quietly = TRUE)) {
  install.packages("ggplot2")
}
library(ggplot2)

# 2. Load dataset
bf <- read.csv("data/bodyfat.csv")

# 3. Inspect and validate data
cat("Dataset dimensions:", dim(bf), "\n")
str(bf)
summary(bf)

cat("\nMissing values by column:\n")
print(colSums(is.na(bf)))
cat("\nDuplicate rows:", sum(duplicated(bf)), "\n")

# Keep only the columns used in this baseline model and remove incomplete rows.
model_data <- bf[, c("Age", "Weight", "Height", "Neck",
                     "Chest", "Abdomen", "Hip", "Thigh",
                     "Knee", "Ankle", "Biceps", "Forearm",
                     "Wrist", "BodyFat")]
model_data <- na.omit(model_data)

# 4. Exploratory Data Analysis

# Plot 1: Body Fat Distribution
p1 <- ggplot(model_data, aes(x = BodyFat)) +
  geom_histogram(
    binwidth = 2,
    fill = "#3498DB",
    colour = "white",
    linewidth = 0.8
  ) +
  labs(
    title = "Distribution of Body Fat Percentage",
    subtitle = "Frequency distribution of body fat values",
    x = "Body Fat (%)",
    y = "Number of Individuals"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(colour = "gray40"),
    panel.grid.minor = element_blank()
  )


# Plot 2: Age vs Body Fat
p2 <- ggplot(model_data, aes(x = Age, y = BodyFat)) +
  geom_point(
    colour = "#2874A6",
    alpha = 0.7,
    size = 2.8
  ) +
  geom_smooth(
    method = "lm",
    se = TRUE,
    colour = "#E74C3C",
    fill = "#F5B7B1",
    linewidth = 1.2
  ) +
  labs(
    title = "Age vs Body Fat Percentage",
    subtitle = "Relationship between age and body fat",
    x = "Age (Years)",
    y = "Body Fat (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(colour = "gray40"),
    panel.grid.minor = element_blank()
  )


# Plot 3: Body Fat Boxplot
p3 <- ggplot(model_data, aes(y = BodyFat)) +
  geom_boxplot(
    fill = "#58D68D",
    colour = "#196F3D",
    width = 0.35,
    alpha = 0.8,
    linewidth = 1
  ) +
  labs(
    title = "Body Fat Percentage – Outlier Analysis",
    subtitle = "Distribution and potential outliers",
    y = "Body Fat (%)",
    x = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(colour = "gray40"),
    panel.grid.minor = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )


# 5. Reproducible train-test split (70% training, 30% testing)
set.seed(123)

train_rows <- sample(
  seq_len(nrow(model_data)),
  size = floor(0.70 * nrow(model_data))
)

train_data <- model_data[train_rows, ]
test_data <- model_data[-train_rows, ]

cat("\nTraining rows:", nrow(train_data), "\n")
cat("Testing rows:", nrow(test_data), "\n")


# 6. Train Multiple Linear Regression
model <- lm(
  BodyFat ~ Age + Weight + Height + Neck +
    Chest + Abdomen + Hip + Thigh + Knee +
    Ankle + Biceps + Forearm + Wrist,
  data = train_data
)

cat("\nModel summary:\n")
print(summary(model))


# 7. Predict on unseen test data
predictions <- predict(model, newdata = test_data)


# 8. Evaluate model
rmse <- sqrt(mean((test_data$BodyFat - predictions)^2))

sst <- sum((test_data$BodyFat - mean(test_data$BodyFat))^2)
sse <- sum((test_data$BodyFat - predictions)^2)

r_squared_test <- if (sst == 0) NA_real_ else 1 - (sse / sst)

cat("\nTest RMSE:", round(rmse, 3), "\n")
cat("Test R-squared:", round(r_squared_test, 3), "\n")


# 9. Actual vs Predicted Plot
prediction_data <- data.frame(
  Actual = test_data$BodyFat,
  Predicted = as.numeric(predictions)
)

p4 <- ggplot(prediction_data, aes(x = Actual, y = Predicted)) +
  geom_point(
    colour = "#2874A6",
    size = 3,
    alpha = 0.8
  ) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = "#E74C3C",
    linewidth = 1.2
  ) +
  labs(
    title = "Actual vs Predicted Body Fat Percentage",
    subtitle = "Multiple Linear Regression – Test Data",
    x = "Actual Body Fat (%)",
    y = "Predicted Body Fat (%)"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(colour = "gray40"),
    panel.grid.minor = element_blank()
  )


# 10. Save Plots
dir.create("visualizations", showWarnings = FALSE)

ggsave("visualizations/bodyfat_distribution.png", p1,
       width = 9, height = 6, dpi = 300)

ggsave("visualizations/age_vs_bodyfat.png", p2,
       width = 9, height = 6, dpi = 300)

ggsave("visualizations/bodyfat_boxplot.png", p3,
       width = 9, height = 6, dpi = 300)

ggsave("visualizations/actual_vs_predicted.png", p4,
       width = 9, height = 6, dpi = 300)

cat("\nPlots saved in the visualizations folder.\n")

print(p1)
print(p2)
print(p3)
print(p4)

cat("\nRMSE:", rmse, "\n")
cat("R-squared:", r_squared_test, "\n")

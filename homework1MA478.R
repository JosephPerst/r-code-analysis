#data explore

#load libs
library(ggplot2)
library(corrplot)
library(dplyr)
library(tidyr)

#load data from working dir
train <- read.csv("moneyball_training.csv")
eval <- read.csv("moneyball_evaluation.csv")

#basic dataset information
cat("Training set:", dim(train)[1], "rows x", dim(train)[2], "columns\n")
cat("Evaluation set:", dim(eval)[1], "rows x", dim(eval)[2], "columns\n\n")

#show available vars
print(names(train))
print(head(train))

#some summary stats
print(summary(train))

#winning percentage stats
cat("\nWinning Percentage stats:\n")
cat("Mean:", mean(train$WP, na.rm = TRUE), "\n")
cat("Median:", median(train$WP, na.rm = TRUE), "\n")
cat("SD:", sd(train$WP, na.rm = TRUE), "\n")


#missing data checks
missing_counts <- colSums(is.na(train))
missing_pct <- (missing_counts / nrow(train)) * 100
missing_df <- data.frame(Variable = names(missing_counts), 
                         Missing_Count = missing_counts,
                         Missing_Percent = round(missing_pct, 2))
missing_df <- missing_df[missing_df$Missing_Count > 0, ]

cat("\nVariables with missing data:\n")
if(nrow(missing_df) > 0) {
  print(missing_df)
} else {
  cat("No missing data\n")
}



#potential correlations with win percentage
numeric_vars <- train %>% select(-yearID, -teamID) %>% select_if(is.numeric)
correlations <- cor(numeric_vars, use = "pairwise.complete.obs")
wp_correlations <- sort(correlations[, "WP"], decreasing = TRUE)

cat("\nCorrelations with Winning Percentage:\n")
print(round(wp_correlations, 3))



#some visualizations

#histograms of winnign percentage
ggplot(train, aes(x = WP)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Winning Percentage") +
  theme_minimal()

#heatmap of correlation
corrplot(correlations, method = "color", type = "upper", tl.col = "black", tl.srt = 45)

#bar chatrt of top correlations
top_cors <- wp_correlations[2:11]  # Top 10, excluding WP itself
cor_df <- data.frame(Variable = names(top_cors), Correlation = as.numeric(top_cors))

ggplot(cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Correlations with WP") +
  theme_minimal()

#HR vs WP
ggplot(train, aes(x = HR, y = WP)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Home Runs vs Winning Percentage") +
  theme_minimal()

#HA vs HP
ggplot(train, aes(x = HA, y = WP)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Hits Allowed vs Winning Percentage") +
  theme_minimal()
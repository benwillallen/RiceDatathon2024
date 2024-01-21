library(tidyverse)
library(xgboost)
library(randomForest)
library(ModelMetrics)
library(fastshap)
library(shapviz)
library(patchwork)
library(caret)
library(modelr)

# Importing Data
set.seed(24)
oil_data <- read_csv("../Data/preprocessed_training_with_estimations2.csv")
oil_data <- oil_data %>% 
  select(-c(total_fluid, proppant_to_frac_fluid_ratio, frac_seasoning, total_proppant, gross_perforated_length))
train_test_split <- read_csv("../Data/StandardSplit.csv")
oil_data_test <- oil_data[-train_test_split$train_test_split,]
oil_data <- oil_data[train_test_split$train_test_split,]

nn_preds <- read_csv("../Data/neural_network_prediction.csv")

# XGBoost Models
# General Model
train_X <- as.matrix(oil_data[,-1])
train_y <- as.matrix(oil_data$OilPeakRate)
xg_model <- xgboost(params=list(verbosity=0, learning_rate=0.1, objective="reg:squarederror", eval_metric = "rmse",
                                max_depth=10, subsample=1, colsample_bytree=0.5, lambda=0.5, alpha=1),
                    data = train_X, label=train_y, nrounds = 80)

# Small Model
small_oil_data <- oil_data %>% 
  filter(OilPeakRate < quantile(oil_data$OilPeakRate, 0.50))
train_X <- as.matrix(small_oil_data[,-1])
train_y <- as.matrix(small_oil_data$OilPeakRate)
small_xg_model <- xgboost(params=list(verbosity=0, learning_rate=0.1, objective="reg:squarederror",
                                      eval_metric = "rmse", max_depth=8, subsample=1, colsample_bytree=0.5,
                                      lambda=0.5, alpha=1),
                          data = train_X, label=train_y, nrounds = 60)

# Large Model
large_oil_data <- oil_data %>% 
  filter(OilPeakRate > quantile(oil_data$OilPeakRate, 0.5))
train_X <- as.matrix(large_oil_data[,-1])
train_y <- as.matrix(large_oil_data$OilPeakRate)
large_xg_model <- xgboost(params=list(verbosity=0, learning_rate=0.1, objective="reg:squarederror",
                                      eval_metric = "rmse", max_depth=8, subsample=1, colsample_bytree=0.5,
                                      lambda=0.5, alpha=1),
                          data = train_X, label=train_y, nrounds = 70)

# Extreme Model
larger_oil_data <- oil_data %>% 
  filter(OilPeakRate > quantile(oil_data$OilPeakRate, 0.9))
train_X <- as.matrix(larger_oil_data[,-1])
train_y <- as.matrix(larger_oil_data$OilPeakRate)
larger_xg_model <- xgboost(params=list(verbosity=0, learning_rate=0.1, objective="reg:squarederror",
                                       eval_metric = "rmse", max_depth=8, subsample=1, colsample_bytree=0.5,
                                       lambda=0.5, alpha=1),
                           data = train_X, label=train_y, nrounds = 40)

# Results on Test Set
test_X <- as.matrix(oil_data_test[,-1])
test_y <- as.matrix(oil_data_test[,1])
xg_preds <- predict(xg_model, newdata=test_X)
small_xg_preds <- predict(small_xg_model, newdata=test_X)
large_xg_preds <- predict(large_xg_model, newdata=test_X)
larger_xg_preds <- predict(larger_xg_model, newdata=test_X)
preds_df <- as_tibble(cbind(xg_preds, small_xg_preds, large_xg_preds, larger_xg_preds, nn_preds))
preds_df <- cbind(preds_df, oil_data_test$OilPeakRate)
colnames(preds_df) <- c(colnames(preds_df)[1:length(preds_df)-1], "OilPeakRate")

preds_X <- as.matrix(preds_df[,-length(preds_df)])
preds_y <- as.matrix(preds_df[,length(preds_df)])

# Blender Model
voter_xg_model <- xgboost(params=list(learning_rate=0.05, objective="reg:squarederror", eval_metric = "rmse",
                                      max_depth=2, subsample=0.5, colsample_bytree=0.75, lambda=1, alpha=10),
                          data = preds_X, label=preds_y, nrounds = 70)

voted_preds <- predict(voter_xg_model, newdata=preds_X)

# RMSE
sqrt(mean((preds_df$xg_preds - preds_df$OilPeakRate)^2))
sqrt(mean((preds_df$small_xg_preds - preds_df$OilPeakRate)^2))
sqrt(mean((preds_df$large_xg_preds - preds_df$OilPeakRate)^2))
sqrt(mean((preds_df$larger_xg_preds - preds_df$OilPeakRate)^2))
sqrt(mean((preds_df$predictions - preds_df$OilPeakRate)^2))
sqrt(mean((voted_preds - preds_df$OilPeakRate)^2))

# Plotting
ggplot(data=preds_df) +
  geom_density(alpha=0.5, mapping=aes(x=small_xg_preds, color="Small XGBoost"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=large_xg_preds, color="Large XGBoost"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=larger_xg_preds, color="Extreme XGBoost"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=xg_preds, color="General XGBoost"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=predictions, color="Dense Neural Network"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=voted_preds, color="Combined Model"), linewidth=2) +
  geom_density(alpha=0.5, mapping=aes(x=OilPeakRate, color="Actual Value"), linewidth=2) +
  theme_dark() +
  theme(plot.background = element_rect(fill = "#211A1E", color="#211A1E"),
        panel.background = element_rect(fill = "#291f25"),
        legend.background = element_rect(fill = "#211A1E"),
        legend.key.height = unit(0.5, "cm"),
        text = element_text(family="Bahnschrift", color="#fff0d5", size=16),
        plot.title = element_text(size=30),
        strip.text = element_text(color="#fff0d5", size=16)) +
  labs(x="Oil Peak Rate", y="Density", title="Prediction Distributions") +
  scale_color_discrete(name="Model")
ggsave("..\Analysis\Visuals\FullPredPlot.png", width=1920, height=1080, units="px")

ggplot(data=preds_df) +
  geom_density(alpha=0.5, mapping=aes(x=voted_preds, color="Combined Model"), linewidth=1) +
  geom_density(alpha=0.5, mapping=aes(x=OilPeakRate, color="Actual Value"), linewidth=1.5) +
  theme_dark() +
  theme(plot.background = element_rect(fill = "#211A1E", color="#211A1E"),
        panel.background = element_rect(fill = "#291f25"),
        legend.background = element_rect(fill = "#211A1E"),
        legend.key.height = unit(0.5, "cm"),
        text = element_text(family="Bahnschrift", color="#fff0d5", size=16),
        plot.title = element_text(size=30),
        strip.text = element_text(color="#fff0d5", size=16)) +
  labs(x="Oil Peak Rate", y="Density", title="Combined Model") +
  scale_color_discrete(name="Model")
ggsave("..\Analysis\Visuals\PredPlot.png", width=1920, height=1080, units="px")

# SHAP Values
explanation <- fastshap::explain(voter_xg_model, X = preds_X, pred_wrapper = predict,
                                 newdata = preds_X, exact=F, nsim=20)
shp <- shapviz(explanation, X=preds_X, baseline=mean(preds_y))
sv_importance(shp, kind = "beeswarm") & theme_bw() 

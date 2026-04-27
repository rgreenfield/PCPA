# set working directory
setwd("/Users/raymondgreenfield/Documents/ACAS/PCPA")
getwd()

#install packages
#install.packages("tidyverse")
library(tidyverse)

# import the data
df <- read.csv("./data/Motorcylcle_claims.csv", header = TRUE)


############################ DATA CLEANING  ####################################

# Clean NA values

df_clean_na <- na.omit(df)

# Check the validity of each field
# Check Age rangea for possible values

min(df_clean_na$Age)
max(df_clean_na$Age)


# Assume only age can go up to 14-99 for the project
df_clean_age <- subset(df_clean_na, 14 <= Age)
df_clean_age <- subset(df_clean_na, Age < 100)

# Check if the geographic zones match the problem statement
unique(df_clean_age$Geographic.Zone)

# Check for MC.Class to be between the values of 1-7
unique(df_clean_age$MC.Class)

# remove the values where MC.Class is 10
df_clean_mc_class <- subset(df_clean_age, MC.Class <= 7)
unique(df_clean_mc_class$MC.Class)

# Check for a valid driving score between 1-7
unique(df_clean_mc_class$Driving.Score)

df_clean_drivers_score <- df_clean_mc_class
df_clean_drivers_score$Driving.Score[df_clean_drivers_score$Driving.Score == 300] <- 3 # clean up the 300 mapping to 3  
unique(df_clean_drivers_score$Driving.Score)

# Check for valid policy length
min(df_clean_drivers_score$Policy.Length)
max(df_clean_drivers_score$Policy.Length)

library(dplyr)

df_clean_drivers_score %>%
  filter(Policy.Length > 100)

# remove the 1 row where the policy_length is < 100
df_clean_policy_length <- df_clean_drivers_score %>%
                              filter(Policy.Length <= 100)

# Check the number of claims
min(df_clean_policy_length$Number.of.Claims)
max(df_clean_policy_length$Number.of.Claims)


# Check claim cost
min(df_clean_policy_length$Claim.Cost)
max(df_clean_policy_length$Claim.Cost)

df <- df_clean_policy_length

# write the cleaned dataset
write.csv(df, file = 'MotoData_Cleaned.csv')


############################ DATA VISUALIZATION  ################################

library(ggplot2)

ggplot(data = dataset) + geom_histogram(aes(variable))
ggplot(data = dataset) + geom_bar(aes(variable))
ggplot(data = dataset) + geom_line(aes(variable))
ggplot(data = dataset) + geom_boxplot(aes(x=variable, y= variable))
ggplot(data = dataset) + geom_point(mapping = aes(x=variable, y=variable))
ggplot(data = dataset) + geom_point(mapping = aes(x= variable, y = variable, color = variable)) + xlab("") + ylab("") + ggtitle("")


ggplot(data = df) + geom_histogram(aes(Claim.Cost), binwidth = 1000)

ggplot(data = df) + geom_boxplot(aes(Claim.Cost))


ggplot(data = df) + geom_histogram(aes(Age), binwidth = 8)
ggplot(data = df) + geom_bar(aes(Geographic.Zone))
ggplot(data = df) + geom_boxplot(aes(x=Claim.Cost, y= Geographic.Zone))

ggplot(data = df) + geom_bar(aes(Driving.Score))

ggplot(data = df) + geom_histogram(aes(Policy.Length))

ggplot(data = df) + geom_point(mapping = aes(x= Age, y = Claim.Cost, color = MC.Class)) + xlab("Age") + ylab("Claim Cost") + ggtitle("Claim Cost vs Age")


# Correlation Matrix
install.packages("corrplot")
library(corrplot)

# drop multiple columns
df_corr <- df[ , !(names(df) %in% c("Geographic.Zone"))]

cor(df_corr)
pairs(df_corr)

corrplot(cor(df_corr), 
         method  = "color",     # or "circle", "number", "ellipse"
         type    = "upper",     # show upper triangle only
         addCoef.col = "black", # overlay correlation numbers
         tl.col  = "black")     # variable name color

############################ DATA PREPARATION #################################
df_cat <- df

df_cat$Northeast <- ifelse(df_cat$Geographic.Zone == "Northeast", 1, 0)
df_cat$East      <- ifelse(df_cat$Geographic.Zone == "East", 1, 0)
df_cat$South     <- ifelse(df_cat$Geographic.Zone == "South", 1, 0)
df_cat$Midwest   <- ifelse(df_cat$Geographic.Zone == "Midwest", 1, 0)
df_cat$West      <- ifelse(df_cat$Geographic.Zone == "West", 1, 0)
df_cat$Southwest <- ifelse(df_cat$Geographic.Zone == "Southwest", 1, 0)
df_cat$Rockies   <- ifelse(df_cat$Geographic.Zone == "Rockies", 1, 0)

# Drop the Geographic.Zone field
df_cat <- df_cat[ , !(names(df_cat) %in% c("Geographic.Zone"))]

view(df_cat)

# Check the datatypes
str(df_cat)

# change datatypes
as.integer()
as.numeric()
as.factor() # categorical field that has levels


# check for outliers
ggplot(data = df_cat) + geom_boxplot(aes(Claim.Cost))
ggplot_build(ggplot(data=df_cat) + geom_boxplot(aes(Claim.Cost)))$data


df_cat_outliers <- subset(df_cat, Claim.Cost < 20764)
ggplot(data = df_cat_outliers) + geom_boxplot(aes(Claim.Cost))


# capping claim cost at 25,000 based on box-plot
df_cat_cap <- df_cat_outliers
df_cat_cap$Claim.Cost <- ifelse(
  df_cat_cap$Claim.Cost > 25000,
  25000,
  df_cat_cap$Claim.Cost
)

ggplot(data = df_cat_cap) + geom_boxplot(aes(Claim.Cost))

# Check for zeros or negatives in Age
sum(df_cat_cap$Age <= 0, na.rm = TRUE)
min(df_cat_cap$Age, na.rm = TRUE)

# Check for NAs
sum(is.na(df_cat_cap$Age))

# Dealing with imbalanced data
library(caret)

# Downsampling
set.seed(123)
#newdataset <- downSample(
#  x= currentdata %>% select(-response variable)
#  ,y = currentdata$responsevariable
#  ,yname = 'Response Variable Name'
#)

# Upsampling
#set.seed(123)
#newdataset <- upSample(
#  x= currentdata %>% select(-response variable)
#  ,y = currentdata$responsevariable
#  ,yname = 'Response Variable Name'
#)


############################ MODEL PREPARATION #################################

set.seed(42)

# train/test split the data (0.7, 0.3)
train_rows <- createDataPartition(df_cat_cap$Claim.Cost, p = 0.7, list = FALSE)

train <- df_cat_cap[train_rows,]
test <- df_cat_cap[-train_rows,]

mean(train$Claim.Cost)
mean(test$Claim.Cost)


############################ MODEL BUILD #################################

# tweedie - zero-inflation data, joint distribution (frequency/severity)
# p = 1-2 (poisson/gamma mix)
# log - link function = appropriate for skewed data, easily interpreted

# Building a model for pure premium = Claim.Cost / Exposure = Claim.Cost / Policy.Length
# Include Policy.Length as offset --> log(Policy.Length) as an offset

# Claim.Cost = Bx + B1 + .. + log(Policy.Length)

library(statmod)
library(tweedie)

# estimating parameter p
est_P <- tweedie_profile(Claim.Cost ~ Age + Vehicle.Age + Driving.Score + MC.Class + Northeast + South + Rockies + East + West + Southwest
                         , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)

# P = 1.6

tm_full <- glm(formula = Claim.Cost ~ Age + Vehicle.Age + Driving.Score + MC.Class + Northeast + South + Rockies + East + West + Southwest
               , data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))


summary(tm_full)

tweedie_AIC(tm_full)
# 6494.213

#AIC(modelname)


# testing for multicollinearity
install.packages("car")
library(car)

vif(tm_full)

# calculate the loglihood
logLiktweedie(tm_full)
# -3235.106

# ITERATION 2 - Significant Variables Only
est_P2 <- tweedie.profile(Claim.Cost ~ Age + MC.Class + Northeast + East
                          , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)

# P=1.6

tm_2 <- glm(formula = (Claim.Cost ~ Age + MC.Class + Northeast + East)
            ,data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))


summary(tm_2)

tweedie_AIC(tm_2)
# 6497.481


# ITERATION 3 - Significant Variables Only
est_P3 <- tweedie.profile(Claim.Cost ~ Age + MC.Class + Northeast
                          , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)

# P=1.6

tm_3 <- glm(formula = (Claim.Cost ~ Age + MC.Class + Northeast)
            ,data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))

summary(tm_3)

tweedie_AIC(tm_3)
# 6623.663


# ITERATION 4 - Adding Transformed variables (Age^2)
train$Age2 <- (train$Age)^2

est_P4 <- tweedie.profile(Claim.Cost ~ Age + MC.Class + Northeast + Age2
                          , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)

# P=1.6

tm_4 <- glm(formula = (Claim.Cost ~ Age + MC.Class + Northeast + Age2)
            ,data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))

summary(tm_4)

AICtweedie(tm_4)

# Remove the Age^2
train <- train[,-c(15)]


# ITERATION 5 - Adding Transformed variables log(Age)

# Removing the rows where Age is 0
train <- train[train$Age != 0, ] 

est_P5 <- tweedie_profile(Claim.Cost ~ log(Age) + MC.Class + Northeast 
                          , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)


tm_5 <- glm(formula = (Claim.Cost ~ log(Age) + MC.Class + Northeast)
            ,data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))

summary(tm_5)

AICtweedie(tm_5)


# ITERATION 6 - Adding Transformed variables (Driving Score^2)

train$DS2 <- (train$Driving.Score)^2

view(train)

# Remove rows where Age is 0, negative, or NA
train <- train[train$Age > 0 & !is.na(train$Age), ]
test  <- test[test$Age > 0 & !is.na(test$Age), ]

est_P6 <- tweedie_profile(Claim.Cost ~ log(Age) + MC.Class + Northeast + Driving.Score + DS2
                          , offset = log(Policy.Length + .01), data = train, link.power = 0, do.plot = TRUE, do.smooth = TRUE)

# P=1.65

tm_6 <- glm(formula = (Claim.Cost ~ log(Age) + MC.Class + Northeast + Driving.Score + DS2)
            ,data = train, family = tweedie(var.power = 1.6, link.power = 0), offset = log(Policy.Length + .01))


summary(tm_6)

tweedie_AIC(tm_6)
# 6604.8

# Compare tm_3 to tm_6

# residuals 
resid_tm3 <- residuals.glm(tm_3, type = "deviance")
resid_tm6 <- residuals.glm(tm_6, type = "deviance")

plot(resid_tm3)
plot(resid_tm6)

plot(train$Age, resid_tm3)
plot(train$Age, resid_tm6)

fit_tm3 <- predict(tm_3)
fit_tm6 <- predict(tm_6)

plot(fit_tm3, resid_tm3)
plot(fit_tm6, resid_tm6)

# residuals are exhibiting larger variance for small claims and smaller variance for larger claims

# QQ plot
qqnorm(resid_tm3)
qqnorm(resid_tm6)

# check test dataset
fit_tm3_test <- predict(tm_3, newdata=test)

test$DS2 <- (test$Driving.Score)^2
fit_tm6_test <- predict(tm_6, newdata=test)

RMSE(fit_tm3, log(train$Claim.Cost + 0.1))
# 5.615562

RMSE(fit_tm3_test, log(test$Claim.Cost + 0.1))
# 6.049334

RMSE(fit_tm6, log(train$Claim.Cost + 0.1))
# 5.541565

RMSE(fit_tm6_test, log(test$Claim.Cost + 0.1))
# 5.749573



# Double Lift Chart
fit_tm3_dl <- predict(tm_3, type = "response")
fit_tm6_dl <- predict(tm_6, type = "response")

doublelift <- data.frame(fit_tm3_dl, fit_tm6_dl, test$Claim.Cost)
doublelift$sortratio <- doublelift$fit_tm3_dl / doublelift$fit_tm6_dl
doublelift$decile <- ntile(doublelift$sortratio, 10)

double_lift_chart <-  aggregate(doublelift$test.Claim.Cost, by = list(doublelift$decile), FUN = mean)

double_lift_chart <- double_lift_chart[, -c(5, 6)]

# long format
install.packages("reshape2")
library(reshape)

dlc <- melt(double_lift_chart, id.vars = "Decile")
ggplot(data = dlc) + geom_line(aes(x=Decile, y=value, color=variable)) + xlab("Decile") + ylab("Average Claim Cost") + ggtitle("Double Lift Chart")

# Both models perform similarly, with tm_6 having a slightly higher average claim cost in the top deciles. 
# This suggests that tm_6 may be better at predicting higher claim costs compared to tm_3.


# Lorenze curl and the Gini Coefficient
install.packages("ineq")
library(ineq)

gini_test <- ineq(test$Claim.Cost, parameter = NULL)
xtitle_test <- paste("Gini Coefficient =", round(gini_test, 4))
plot(Lc(test$Claim.Cost), col = "blue", lwd = 2, main = "Lorenz Curve", xlab = xtitle_test)







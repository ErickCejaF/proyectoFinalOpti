head(Boston)

##################################################################
##################################################################
# Simple linear regression

# First part
lm.fit <- lm(medv ~ lstat, data = Boston)
attach(Boston)
lm.fit <- lm(medv ~ lstat)
summary(lm.fit)

lm(formula = medv ~ lstat)
names(lm.fit)
coef(lm.fit)
confint(lm.fit)

# Predictions
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "confidence")
predict(lm.fit, data.frame(lstat = (c(5, 10, 15))), interval = "prediction")

# Plots
plot(lstat, medv)
abline(lm.fit)

# adding pretty things
abline(lm.fit, lwd = 3)
abline(lm.fit, lwd = 3, col = " red ")

plot(lstat, medv, col = " red ")
plot(lstat, medv, pch = 20)
plot(lstat, medv, pch = "+")
plot(1:20, 1:20, pch = 1:20)


par(mfrow = c(2, 2))
plot(lm.fit)

plot(predict(lm.fit), residuals(lm.fit))
plot(predict(lm.fit), rstudent(lm.fit))

plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit))

##################################################################
##################################################################
#  Multiple linear regression

lm.fit <- lm(medv ~ lstat + age, data = Boston)
summary(lm.fit)

lm.fit <- lm(medv ~ ., data = Boston)
summary(lm.fit)
vif(lm.fit)

lm.fit1 <- lm(medv ~ . - age, data = Boston)
summary(lm.fit1)
lm.fit1 <- update(lm.fit, ~. - age)

##################################################################
##################################################################
#  Interaction terms

summary(lm(medv ~ lstat * age, data = Boston))


##################################################################
##################################################################
# Non-linear Transformations of the Predictors

lm.fit2 <- lm(medv ~ lstat + I(lstat^2))
summary(lm.fit2)

lm.fit <- lm(medv ~ lstat)
anova(lm.fit, lm.fit2)

par(mfrow = c(2, 2))
plot(lm.fit2)

lm.fit5 <- lm(medv ~ poly(lstat, 5))
summary(lm.fit5)

summary(lm(medv ~ log(rm), data = Boston))


##################################################################
##################################################################
# Qualitative Predictors

head(Carseats)

lm.fit <- lm(Sales ~ . + Income:Advertising + Price:Age, data = Carseats)
summary(lm.fit)

attach(Carseats)
contrasts(ShelveLoc)


##################################################################
##################################################################
# Functions

LoadLibraries <- function() {
  library(ISLR2)
  library(MASS)
  library(car)
  print("The libraries have been loaded .")
}



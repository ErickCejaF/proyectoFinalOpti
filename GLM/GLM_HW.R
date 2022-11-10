library(ISLR2)
library(MASS)
library(car)

############################################################
############################################################
# 3.8

# a)
lm.fit <- lm(mpg ~ horsepower, data = Auto)
attach(Auto)
lm.fit <- lm(mpg ~ horsepower)
summary(lm.fit)

lm(formula = mpg ~ horsepower)

# b)
plot(mpg, horsepower)
abline(lm.fit)

# adding pretty things
abline(lm.fit, lwd = 3)
abline(lm.fit, lwd = 3, col = " red ")

plot(mpg, horsepower, col = " red ")
plot(mpg, horsepower, pch = 20)
plot(mpg, horsepower, pch = "+")
plot(1:20, 1:20, pch = 1:20)


par(mfrow = c(2, 2))
plot(lm.fit)

print("hola Mundo")

############################################################
############################################################


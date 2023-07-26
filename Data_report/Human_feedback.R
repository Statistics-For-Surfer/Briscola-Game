rm(list = ls())
library(readr)
report <- read_csv("Pausetta Briscola? (Risposte) - Answers.csv")
livs <- report$`How would you rate your Briscola skills?`
results <- report$`Which was your final score?`
mean(results[livs <= 3])
mean(results[livs > 3])




a <- mean(results[livs <= 3])
b <- mean(results[livs > 3])

livs[livs <= 3] <- 0
livs[livs > 3] <- 1




# Plots
library(ggplot2)


livs <- c("Discrete level", "High level")
#m <- c(rep("1-3", floor(a)) , rep("4-5", floor(b)))
m <- c(a,b)
data <- data.frame(livs, m)

data$livs <- factor(data$livs, levels = c("Discrete level", "High level"))
custom_colors <- c("Discrete level" = "#118ab2", "High level" = "#ffd166")

ggplot(data, aes(y=m, x=livs , fill = livs)) + 
  geom_bar(position="dodge", stat="identity") + 
  ylim(c(0,115)) +
  xlab("") +
  scale_fill_manual(values = custom_colors) + 
  ylab("Average points") +
  guides(fill = guide_legend(title="Oppoents Level")) +
  theme_light()
  






decks_10 <- data.frame(matrix(c(773, 213, 14, 937, 63, 0), nrow=2, ncol = 3, byrow = T))
colnames(decks_10) <- c('Wins', 'Loss', 'Ties')
rownames(decks_10) <- c('Random', 'Greedy')
decks_10

decks_20 <- data.frame(matrix(c(783, 207, 10, 799, 201, 0), nrow=2, ncol = 3, byrow = T))
colnames(decks_20) <- c('Wins', 'Loss', 'Ties')
rownames(decks_20) <- c('Random', 'Greedy')
decks_20

decks_50 <- data.frame(matrix(c(658, 330, 12, 604, 378, 18), nrow=2, ncol = 3, byrow = T))
colnames(decks_50) <- c('Wins', 'Loss', 'Ties')
rownames(decks_50) <- c('Random', 'Greedy')
decks_50

decks_100 <- data.frame(matrix(c(632, 351, 17, 494, 495,  11), nrow=2, ncol = 3, byrow = T))
colnames(decks_100) <- c('Wins', 'Loss', 'Ties')
rownames(decks_100) <- c('Random', 'Greedy')
decks_100

decks_200 <- data.frame(matrix(c(555, 423, 22, 414, 556, 30), nrow=2, ncol = 3, byrow = T))
colnames(decks_200) <- c('Wins', 'Loss', 'Ties')
rownames(decks_200) <- c('Random', 'Greedy')
decks_200

decks_inf <- data.frame(matrix(c(513, 471, 16, 317, 668, 15), nrow=2, ncol = 3, byrow = T))
colnames(decks_inf) <- c('Wins', 'Loss', 'Ties')
rownames(decks_inf) <- c('Random', 'Greedy')
decks_inf




# Plots
library(ggplot2)


Decks <- c(rep("10", 2), rep("20", 2), rep("50", 2), rep('100', 2), rep("200", 2), rep('all', 2))
Opponent <- rep(c("Random", "Greedy"), 6)
Win <- c(0.773, 0.937, 0.783, 0.799, 0.658, 0.604, 0.632, 0.494, 0.555, 0.414, 0.513, 0.317)
data <- data.frame(Decks, Opponent, Win)
data$Decks <- factor(data$Decks, levels = c("10", "20", "50", "100", "200", "all"))
data$Opponent <- factor(data$Opponent, levels = c("Random", "Greedy"))
custom_colors <- c("Random" = "#118ab2", "Greedy" = "#ffd166")


ggplot(data, aes(fill=Opponent, y=Win, x=Decks)) + 
  geom_bar(position="dodge", stat="identity") + 
  ylim(c(0,1)) +
  scale_fill_manual(values = custom_colors) + 
  ylab('Win Ratio')


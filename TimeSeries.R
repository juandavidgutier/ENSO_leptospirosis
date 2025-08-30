library(dplyr)

# LOAD THE DATASET
muni_Col <- read.csv("D:/clases/UDES/fortalecimiento institucional/macroproyecto_2023/articulo2/leptos/data_ts.csv")
dim(muni_Col)
head(muni_Col)

# Fig. 2
ts_20muni <- muni_Col %>%
  group_by(period) %>%
  summarise(total_cases = sum(Cases))

Cases <- ts(ts_20muni$total_cases, start = c(2007,1), frequency = 12)

plot(Cases, col="red", xaxt="n")  

years_to_show <- seq(2007, 2024, by = 2)
positions <- (years_to_show - 2007) * 12 + 1

max_pos <- length(Cases)
positions <- positions[positions <= max_pos]
years_to_show <- years_to_show[1:length(positions)]
axis(1, at = time(Cases)[positions], labels = years_to_show, cex.axis = 0.8)
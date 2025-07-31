library(dplyr)

# LOAD THE DATASET
muni_Col <- read.csv("D:/data_ts.csv")
dim(muni_Col)
head(muni_Col)

# Fig. time series
ts_20muni <- muni_Col %>%
  group_by(period) %>%
  summarise(total_cases = sum(Cases))

Cases <- ts(ts_20muni$total_cases, start = c(2007,1), frequency = 12)
plot(Cases, col="red")

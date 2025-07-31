library(ggdag)
library(dagitty)
library(lavaan)
library(CondIndTests)
library(dplyr)
library(GGally)
library(tidyr)
library(MKdescr)
library(caret)

# Load data
dataset <- read.csv("D:/data_DAG.csv")
dataset <- select(dataset, excess, S3, S4, S34, S12, ESOI, SOI, 
                  NATL, SATL, TROP, MPI, sro, stl1, swvl1, t2m, tp, 
                  pop_density, NeutralvsLa_Nina)
str(dataset)

# z-scores
dataset$S12 <- zscore(dataset$S12, na.rm = TRUE)  
dataset$S3 <- zscore(dataset$S3, na.rm = TRUE) 
dataset$S34 <- zscore(dataset$S34, na.rm = TRUE) 
dataset$S4 <- zscore(dataset$S4, na.rm = TRUE)
dataset$ESOI <- zscore(dataset$ESOI, na.rm = TRUE)  
dataset$SOI <- zscore(dataset$SOI, na.rm = TRUE) 
dataset$NATL <- zscore(dataset$NATL, na.rm = TRUE) 
dataset$SATL <- zscore(dataset$SATL, na.rm = TRUE) 
dataset$TROP <- zscore(dataset$TROP, na.rm = TRUE)
dataset$MPI <- zscore(dataset$MPI, na.rm = TRUE) 
dataset$sro <- zscore(dataset$sro, na.rm = TRUE)  
dataset$stl1 <- zscore(dataset$stl1, na.rm = TRUE)  
dataset$swvl1 <- zscore(dataset$swvl1, na.rm = TRUE) 
dataset$t2m <- zscore(dataset$t2m, na.rm = TRUE) 
dataset$tp <- zscore(dataset$tp, na.rm = TRUE)
dataset$pop_density <- zscore(dataset$pop_density, na.rm = TRUE) 

# Drop NAs
dataset <- dataset[complete.cases(dataset), ]
str(dataset)

# DAG
dag <- dagitty('dag {
excess [pos="1, 0.5"]
NeutralvsLa_Nina [pos="-1, 0.5"]
S12 [pos="-1.4, 1.1"]
S3 [pos="-1.8, 1.3"]
S34 [pos="-2, 1.5"]
S4 [pos="-1.9, 1.4"]
SOI [pos="-1.6, 1.1"]
ESOI [pos="-2.6, 2.1"]
NATL [pos="-2.2, 1.7"]
TROP [pos="-2.4, 1.9"]
SATL [pos="-2.8, 2.3"]

sro [pos="-1.0, -2.5"] 
stl1 [pos="-1.1, -2.6"] 
swvl1 [pos="-1.2, -2.7"]
t2m [pos="-1.3, -2.8"]
tp [pos="-1.4, -2.9"] 
MPI [pos="0.5, -1.25"]
pop_density [pos="0.5, 1.25"]

S12 -> S3 S12 -> S34 S12 -> S4 S12 -> SOI S12 -> ESOI S12 -> NATL S12 -> SATL S12 -> TROP
S3 -> S34 S3 -> S4 S3 -> SOI S3 -> ESOI S3 -> NATL S3 -> SATL S3 -> TROP
S34 -> S4 S34 -> SOI S34 -> ESOI S34 -> NATL S34 -> SATL S34 -> TROP
S4 -> SOI S4 -> ESOI S4 -> NATL S4 -> SATL S4 -> TROP
SOI -> ESOI SOI -> NATL SOI -> SATL SOI -> TROP
ESOI -> NATL ESOI -> SATL ESOI -> TROP
NATL -> SATL NATL -> TROP
SATL -> TROP

S12 -> NeutralvsLa_Nina S3 -> NeutralvsLa_Nina S34 -> NeutralvsLa_Nina S4 -> NeutralvsLa_Nina
SOI -> NeutralvsLa_Nina ESOI -> NeutralvsLa_Nina NATL -> NeutralvsLa_Nina SATL -> NeutralvsLa_Nina
TROP -> NeutralvsLa_Nina

S12 -> excess S3 -> excess S34 -> excess S4 -> excess SOI -> excess ESOI -> excess
NATL -> excess SATL -> excess TROP -> excess

S12 -> sro S3 -> sro S34 -> sro S4 -> sro SOI -> sro ESOI -> sro NATL -> sro SATL -> sro TROP -> sro
S12 -> stl1 S3 -> stl1 S34 -> stl1 S4 -> stl1 SOI -> stl1 ESOI -> stl1 NATL -> stl1 SATL -> stl1 TROP -> stl1
S12 -> swvl1 S3 -> swvl1 S34 -> swvl1 S4 -> swvl1 SOI -> swvl1 ESOI -> swvl1 NATL -> swvl1 SATL -> swvl1 TROP -> swvl1
S12 -> t2m S3 -> t2m S34 -> t2m S4 -> t2m SOI -> t2m ESOI -> t2m NATL -> t2m SATL -> t2m TROP -> t2m
S12 -> tp S3 -> tp S34 -> tp S4 -> tp SOI -> tp ESOI -> tp NATL -> tp SATL -> tp TROP -> tp


NeutralvsLa_Nina -> sro NeutralvsLa_Nina -> stl1 NeutralvsLa_Nina -> swvl1 NeutralvsLa_Nina -> t2m NeutralvsLa_Nina -> tp NeutralvsLa_Nina -> excess

sro -> MPI stl1 -> MPI swvl1 -> MPI t2m -> MPI tp -> MPI
sro -> excess stl1 -> excess swvl1 -> excess t2m -> excess tp -> excess
sro -> pop_density stl1 -> pop_density swvl1 -> pop_density t2m -> pop_density tp -> pop_density


sro -> stl1 sro -> swvl1 sro -> t2m sro -> tp
stl1 -> swvl1 stl1 -> t2m stl1 -> tp
swvl1 -> t2m swvl1 -> tp
t2m -> tp

MPI -> excess MPI -> pop_density
pop_density -> excess
}')

# Plot DAG
plot(dag)

# Covariance matrix
myCov <- cov(dataset)
round(myCov, 2)

# Colineality
myCor <- cov2cor(myCov)
noDiag <- myCor
diag(noDiag) <- 0
any(noDiag == 1)  

# Multicolineality
det(myCov) < 0
any(eigen(myCov)$values < 0)

# Conditional independeces
impliedConditionalIndependencies(dag)
corr <- lavCor(dataset)

# LocalTests
localTests(dag, sample.cov = myCov, sample.nobs = nrow(dataset))

# Plot 
plotLocalTestResults(localTests(dag, sample.cov=corr, sample.nobs=nrow(dataset)), xlim=c(-1,1))


# Identification
simple_dag <- dagify(
  excess ~  NeutralvsLaNina + S12 + S3 + S34 + S4 + SOI + ESOI + NATL + SATL + TROP + sro + stl1 + swvl1 + t2m + tp + MPI + pop_density,
  NeutralvsLaNina ~ S12 + S3 + S34 + S4 + SOI + ESOI + NATL + SATL + TROP,
  S12 ~ S3 + S34 + S4 + SOI + ESOI + NATL + SATL + TROP,
  S3 ~ S34 + S4 + SOI + ESOI + NATL + SATL + TROP,
  S34 ~ S4 + SOI + ESOI + NATL + SATL + TROP,
  S4 ~ SOI + ESOI + NATL + SATL + TROP,
  SOI ~ ESOI + NATL + SATL + TROP,
  ESOI ~ NATL + SATL + TROP,
  NATL ~ SATL + TROP,
  SATL ~ TROP,
  
  
  sro ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + stl1 + swvl1 + t2m + tp,
  stl1 ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + sro + swvl1 + t2m + tp,
  swvl1 ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + sro + stl1 + t2m + tp,  
  t2m ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + sro + stl1 + swvl1 + tp, 
  tp ~ NeutralvsLaNina + S3 + S34 + S4 + SOI + NATL + TROP + sro + stl1 + swvl1 + t2m,   
  
  MPI ~ sro + stl1 + swvl1 + t2m + tp,
  pop_density ~ MPI + sro + stl1 + swvl1 + t2m + tp,
  exposure = "NeutralvsLaNina",
  outcome = "excess",
  coords = list(x = c(excess=1, NeutralvsLaNina=-1, ESOI=-1.6, SOI=-1.7, S3=-1.8, S4=-1.9, S34=-2, NATL=-2.1, TROP=-2.2, S12=-2.3, SATL=-2.4,
                      sro=-1.0, stl1=-0.9, swvl1=-0.8, t2m=-0.7, tp=-0.6,
                      MPI=0.5, pop_density=0.5),
                y = c(excess=0.5, NeutralvsLaNina=0.5, ESOI=1.0, SOI=1.1, S3=1.2, S4=1.3, S34=1.4, NATL=1.5, TROP=1.6, S12=1.7, SATL=1.8,
                      sro=-2.5, stl1=-2.4, swvl1=-2.3, t2m=-2.2, tp=-2.1,
                      MPI=-1.25, pop_density=1.25))
)

ggdag(simple_dag) + 
  theme_dag()

ggdag_status(simple_dag) +
  theme_dag()

# Adjusting
adjustmentSets(simple_dag,  type = "minimal")

ggdag_adjustment_set(simple_dag, shadow = TRUE) +
  theme_dag()



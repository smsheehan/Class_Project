library(tidyverse)

perc_total_homeless_vf <- read.csv('Percent_Total_Homeless_vf.csv',check.names = F,stringsAsFactors = F)

lm(Unsheltered_perc_tot~TotalBeds_perc_tot+Emergency_perc_tot+Transitional_perc_tot+SafeHaven_perc_tot+PermanentSupportive_perc_tot+PermanentOther_perc_tot+RapidRehousing_perc_tot, data = perc_total_homeless_vf)

summary(lm(Unsheltered_perc_tot~+TotalBeds_perc_tot+Emergency_perc_tot+Transitional_perc_tot+SafeHaven_perc_tot+PermanentSupportive_perc_tot+PermanentOther_perc_tot+RapidRehousing_perc_tot, data = perc_total_homeless_vf))

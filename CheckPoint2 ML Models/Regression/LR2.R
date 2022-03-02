library(tidyverse)
all_years_unshelt <- read.csv('AllYears_converted.csv',check.names = F,stringsAsFactors = F)
lm(Unsheltered_perc_pop~Population+Unemployment+ES_beds_perc_pop+TS_beds_perc_pop+SH_beds_perc_pop+RRH_units_perc_pop+PSH_units_perc_pop+OPH_units_perc_pop, data = all_years_unshelt)
summary(lm(Unsheltered_perc_pop~Population+Unemployment+ES_beds_perc_pop+TS_beds_perc_pop+SH_beds_perc_pop+RRH_units_perc_pop+PSH_units_perc_pop+OPH_units_perc_pop, data = all_years_unshelt))

all_years_perc_total_homeless <- read.csv('Percent_Total_Homeless_DroppedColumns_AllCoC.csv',check.names = F,stringsAsFactors = F)
lm(Unsheltered_perc_tot~Population+Unemployment+TotalBeds_perc_tot+Emergency_perc_tot+Transitional_perc_tot+SafeHaven_perc_tot+PermanentSupportive_perc_tot+PermanentOther_perc_tot+RapidRehousing_perc_tot, data = all_years_perc_total_homeless)
summary(lm(Unsheltered_perc_tot~Population+Unemployment+TotalBeds_perc_tot+Emergency_perc_tot+Transitional_perc_tot+SafeHaven_perc_tot+PermanentSupportive_perc_tot+PermanentOther_perc_tot+RapidRehousing_perc_tot, data = all_years_perc_total_homeless))

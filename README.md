# Class_Project: Exploring factors that lead to improved homelessness outcomes

## Presentation
![image](https://user-images.githubusercontent.com/90977689/152702817-a3d32ef1-dd51-44df-b740-ed84e744625b.png)

The US government's Department of Housing and Urban Development (HUD) funds the continuum of care (COC) in major cities and select rural areas across the USA.

![image](https://user-images.githubusercontent.com/90977689/152705248-1869b941-8bc6-4ac6-a999-4e86ff8537e6.png)


As a requirement for receiving funding, every COC must conduct a Point In Time (PIT) count.  One night every year a group of service providers and volunteers in each city surveys and counts the homeless individuals and families.  Additionally, COCs must provide a Housing Inventory Count (HIC) and maintain a Homeless Management Information System (HMIS) database.  While the HMIS database is not open to the public, HUD makes available the PIT and HIC data from across the country.  https://www.huduser.gov/portal/datasets/ahar/2020-ahar-part-1-pit-estimates-of-homelessness-in-the-us.html

Nationwide, there is a disturbing rising trend in the percentage of unsheltered homeless.

![image](https://user-images.githubusercontent.com/90977689/152703171-8a150b3b-e5df-499b-b94a-bc41e8134147.png)

We also see this trend in Indianapolis, especially in 2021.  Here in Indianapolis, the agency known as CHIP (Coalition for Homelessness Intervention and Prevention) is the coordinating entity to ensure our city meets HUD reporting requirements.  Some information on the 2021 PIT cound here in Indy can be found at:  https://www.chipindy.org/reports.html

![image](https://user-images.githubusercontent.com/90977689/152703473-3930823d-0c9a-4ebd-871a-85a0d1556076.png)

This project will use the HUD nationwide PIT and HIC datasets for COCs in the United States combined with population and homelessness funding allocation data to evaluate if machine learning models can identify factors (investment in low barrier shelter beds, investment in housing first units, federal funding levels, etc) which lead to better outcomes (lower homelessness as a percentage of city population).

2020 PIT dataframe:
![image](https://user-images.githubusercontent.com/90977689/152704842-65669730-8a37-4b01-9f6b-a75b2eec8e42.png)

2020 HIC dataframe:
![image](https://user-images.githubusercontent.com/90977689/152704941-6cf31b50-3949-420b-b676-72d0bfd4bd76.png)

Additional datasets that will be pulled into this analysis include HUD funding levels at the programmatic level accross the COCs and population data available from the US Census.  The hypothesis for this project is that we will be able to correlate certain factors across these broad nation-wide data sets with homelessness outcomes.  Successful homelessness outcomes for this project will be defined by cities with the lowest percent of unsheltered and/or unsheltered homeless as a percentage of the city's population.  These findings would then be the first step in helping to guide additional dialog and potentially new investments in our city of Indianapolis.  Currently the Mayor of Indianapolis has a proposal to spend 12.5 million dollars on a new low barrier shelter which would include additional transitional beds. Does our modeling provide evidence in support of this proposal?

## Machine learning Model

The overall strategy for our machine learning approach is to first establish if there are key features from the combined PIT/HIC datasets which are correlated with the outcome of unsheltered individuals.  We felt like random forrest models would provide an initial visualization of feature importances.  If there were strong correlations with outcomes, we hoped there might be a slim chance that we would be able to build a predictive deep learning model to predict if changing the number of beds/units of certain types in a given city would be predicted to deliver an improvement in the unsheltered individuals target.  Since the CoC's represent areas of greatly differing populations, we anticipated that we would need some way to normalize across CoCs.  We envisioned that transforming our data into percents of total CoC population would be a reasonable way to do this.

### Models based on data as a percentage of total CoC region population

The first step was to connect the model to our SQL lite database to pull the data for all the years up through 2019:

![image](https://user-images.githubusercontent.com/90977689/155895236-dadb0e91-693a-48a8-9a55-32069ae25a55.png)

After dropping some unwanted columns, the data needed to be transformed into percent of population.  Exemplar code is shown below:

![image](https://user-images.githubusercontent.com/90977689/155895318-fb684a21-df6f-4b6e-b03d-f23191d7eb3c.png)

After dropping the original columns, the data frame was ready for initial exploration:

![image](https://user-images.githubusercontent.com/90977689/155895403-4d32a11e-77e7-49a4-b392-7f95b0b9bbda.png)

Using the Random Forrest Regressor from sklearn, "Unsheltered_perc_pop" was selected as the target.  The train, test, split data was not scaled for this model.  Results are shown below:

![image](https://user-images.githubusercontent.com/90977689/155895511-ed9567a6-78fd-4de5-9da8-9857f73117ff.png)

![image](https://user-images.githubusercontent.com/90977689/155895543-1e239004-27fc-4019-bfbf-e811a8bec830.png)

![image](https://user-images.githubusercontent.com/90977689/155895570-d6d85652-9f59-4b60-bd03-40d3a01d932f.png)

This exercise was also performed for the data on individual years with some feature importances changing from year to year.  These notebook files are available in the ML folder.  The next question is whether this data is good enough to be used in predictive fashion. A neural network model was build using tensor flow and evidently the data is not strong enough to create a useful model of this type.  Preprocessing for this model included breaking down the "Unsheltered_perc_pop" category into quantiles using the following code:

q = df_AllYears['Unsheltered_perc_pop'].quantile(np.arange(10) / 10)
df_AllYears['UnshelteredPercentQuantile'] = df_AllYears['Unsheltered_perc_pop'].apply(lambda x : q.index[np.searchsorted(q, x, side='right')-1])

The original "Unsheltered_perc_pop" column was dropped and the NN  model was run with the quantile being the target.  The feature data was scaled using X-scaler:

![image](https://user-images.githubusercontent.com/90977689/155895876-f845cb5d-e310-4183-baa3-3936103503ed.png)

![image](https://user-images.githubusercontent.com/90977689/155895896-8bc0a1e7-e3ec-419a-a56a-c9a9fdb795e9.png)

Unfortunately, this resulted in a low accuracy model:

![image](https://user-images.githubusercontent.com/90977689/155895932-82f8554a-30ff-43d3-9e3c-909f6eb33fee.png)

This suggests that perhaps calibrating our data to region population is not a meaningful approach since there may be many other factors that drive percent outcomes of population (for example average temperature or local policing policies).  Before trying a different calibration method, this data was evaluated using a regression model using R studio:

![image](https://user-images.githubusercontent.com/90977689/155896065-b52fb132-c748-4d5d-b9fb-287f5a5185bd.png)

Looking at the data of one of the features with the strongest level of significance (unemployment) we see that the trend is a bit iffy and probably even that is driven by a few major outliers.
![image](https://user-images.githubusercontent.com/90977689/155899652-c9b8ba64-f376-4970-9baf-776824f4ee31.png)

### Models based on data as a percentage of total homeless population





## Database
The extracted dataframes from the HUD data will be stored utilizing a SQlite database. Once data is cleaned of irrelevant columns and cleaned to eliminate non-populated data and revise any pieces of data that diverge from the expected value ranges, the tables will be merged based upon the CoC indices as our master key for all values. Because the content of our database is expected to be manageable from a data size standpoint, all files will be saved to the repository, with the ability to read in the data from any project participant individual on their own machine, reducing the need to rely upon a centrally stored database. This will also serve to let each team member manipulate the data on their unique branches to ensure that all content is sercure, yet able to accomodate independent research without burdensome lockout/tag-out procedures to maintain data integrity. The tables we intend to read into the dataframe include: <br>
- CityData
- ShelterAvailability
- HomelessCounts
- HomelessFunding

![Database Diagram](https://user-images.githubusercontent.com/81537476/155765382-7224ef01-5310-4647-b64b-920b4d6a4b3a.png)



If there is additional need to contextualize the data, we intend to incorporate [metropolitan statistical area data](https://www.census.gov/programs-surveys/metro-micro/about.html) that can assist in providing some demographic and socioeconomic context to the areas supported by the specific CoCs of interest in our analysis.

## Summary

## Comments

## Communication Protocols
- As a group we will be using Slack for a majority of our communication especially during class time and outside of it as well.
- Meeting as a team outside of class hours will take place via Zoom.
- We will be checking out Trello for potential project planning communication.

## Link to Tableau Dashboard
[link to dashboard](https://public.tableau.com/app/profile/michal.upchurch/viz/HICClassProject/TopHomelessPopulationbyCoC?publish=yes "link to dashboard")

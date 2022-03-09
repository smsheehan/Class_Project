# Class_Project: Exploring factors that lead to improved unsheltered homelessness outcomes

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

Additional datasets that will be pulled into this analysis include HUD funding levels at the programmatic level accross the COCs and population data available from the US Census.  County-level and state-level data that is able to be translated into CoC-level units will be integrated to provide context beyond HUD information. This will include population, poverty level, and geolocation data that will allow for a more robust analysis and presentation of data at both the CoC level as well as by census unit. This data will be combined and standardized using FIPS coding to keep consistent across all data sources, which will allow indexing to each HUD CoC.

The hypothesis for this project is that we will be able to correlate certain factors across these broad nation-wide data sets with homelessness outcomes.  Successful homelessness outcomes for this project will be defined by cities with the lowest percent of unsheltered and/or unsheltered homeless as a percentage of the city's population.  These findings would then be the first step in helping to guide additional dialog and potentially new investments in our city of Indianapolis.  Currently the Mayor of Indianapolis has a proposal to spend 12.5 million dollars on a new low barrier shelter which would include additional transitional beds. Does our modeling provide evidence in support of this proposal?

## Machine learning Model

The overall strategy for our machine learning approach is to first establish if there are key features from the combined PIT/HIC datasets which are correlated with the outcome of unsheltered individuals.  We felt like random forest models would provide an initial visualization of feature importances.  If there were strong correlations with outcomes, we hoped there might be a slim chance that we would be able to build a predictive deep learning model to predict if changing the number of beds/units of certain types in a given city would be predicted to deliver an improvement in the unsheltered individuals target.  Since the CoC's represent areas of greatly differing populations, we anticipated that we would need some way to normalize across CoCs.  We envisioned that transforming our data into percents of total CoC population would be a reasonable way to do this.
### Initial Modeling Attempts:
### Models based on data as a percentage of total CoC region population

The first step was to connect the model to our SQL lite database to pull the data for all the years up through 2019:

![image](https://user-images.githubusercontent.com/90977689/155895236-dadb0e91-693a-48a8-9a55-32069ae25a55.png)

After dropping some unwanted columns, the data needed to be transformed into percent of population.  Exemplar code is shown below:

![image](https://user-images.githubusercontent.com/90977689/155895318-fb684a21-df6f-4b6e-b03d-f23191d7eb3c.png)

After dropping the original columns, the data frame was ready for initial exploration:

![image](https://user-images.githubusercontent.com/90977689/155895403-4d32a11e-77e7-49a4-b392-7f95b0b9bbda.png)

Using the Random Forest Regressor from sklearn, "Unsheltered_perc_pop" was selected as the target.  The train, test, split data was not scaled for this model.  Results are shown below:

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

Looking at the data of one of the features with the strongest level of significance (unemployment) we see that the trend is a bit iffy and probably even that is driven by a few major outliers.  The below visualization shows the quintiles of unsheltered as a percent of population where quintile 0.0 is the lowest percentage quintile and quintile 0.9 being the highest percentage quintile.

![image](https://user-images.githubusercontent.com/90977689/155899652-c9b8ba64-f376-4970-9baf-776824f4ee31.png)

### Models based on data as a percentage of total number of homeless

The abover results based on total population caused a rething of how to calibrate the data.  Another potential avenue is to calibrate our numbers relative to the total number of homeless.  Using a similar approach as above, a random forest model was built:

![image](https://user-images.githubusercontent.com/90977689/155900193-5f58f88d-2d7d-4867-8cc1-fa939d63b1df.png)

transform using total number of homeless:

![image](https://user-images.githubusercontent.com/90977689/155900238-8e821ccf-2e04-47c9-9205-b55f484619e4.png)

Build and run the model as before:

![image](https://user-images.githubusercontent.com/90977689/155900274-104c10a4-2b3c-4d0a-a44a-c54bbf110bef.png)

Feature importances:

![image](https://user-images.githubusercontent.com/90977689/155900334-800fe4b4-0ece-4a27-acda-6f651b28b53d.png)

We see Total Beds as a percentage of the total number of homeless to be the strongest feature, which makes sense.  Note total beds wasn't included in the percent population model, since it is just the sum of the ES, TS, and SH bed columns.

![image](https://user-images.githubusercontent.com/90977689/155900414-04d6f2ab-7c13-4ca6-b1a0-7c2a4996bae8.png)

Running a regression model in R shows us that more of the features now achieve significance:

![image](https://user-images.githubusercontent.com/90977689/155900459-5c951bef-773f-4cb1-9013-b286431d9731.png)

However, applying this approach to our NN model does not really improve the accuracy of our model:

![image](https://user-images.githubusercontent.com/90977689/155900530-2958a065-2e92-4157-a87f-8814ecd0ff89.png)

Clearly this speaks to the noise within the data set.  PIT counts are not homogeneous in the methodology used across CoCs and are counting individuals usually on a single night (sometimes two nights).  Next steps to getting to models which can be used prospectively will involve further evaluation and processing of the data to determine a path forward.  This may be a case where some of the older data in the set is less relevant (older methodologies) than more recent data and as a result is adding noise to the set. On the to do list is to use select years (2015-2018) as the training set and year 2019 as the test set. Additionally it may be that rural CoC data may be dramatically distinct from Urban CoCs so we may have to break the data down further to improve our predictivity.  Some high level data looking at trends in total homeless and total unsheltered across CoC category which highlights some differences between the regional CoC types:

![image](https://user-images.githubusercontent.com/90977689/155900789-f44a31e1-c1df-4e8f-85cc-678b6bd4589b.png)

### Final ML Solutions:
![image](https://user-images.githubusercontent.com/90977689/156936700-78638593-a1ee-448a-89b6-c8450d71f931.png)

![image](https://user-images.githubusercontent.com/90977689/156946233-6c6b0d16-ddd8-454e-9d71-cbdef78090ca.png)

![image](https://user-images.githubusercontent.com/90977689/156936825-063ae732-564b-48eb-8613-c6976264448a.png)

![image](https://user-images.githubusercontent.com/90977689/156936840-a202786c-1df9-4424-9b56-f15b91a12697.png)

![image](https://user-images.githubusercontent.com/90977689/156936861-5da2016d-29b2-4c6b-9015-033bb5ae6d8c.png)

![image](https://user-images.githubusercontent.com/90977689/156936882-3556c956-434e-4011-b8e0-223d71d3086c.png)

![image](https://user-images.githubusercontent.com/90977689/156936963-f7cd5112-2112-4676-b8d5-1790780e2805.png)

![image](https://user-images.githubusercontent.com/90977689/156936998-d3a76795-a0d6-4b78-8640-f4c42ead994c.png)

![image](https://user-images.githubusercontent.com/90977689/156937055-9cf4020e-9565-43fd-8251-8b2097a1d9ec.png)

![image](https://user-images.githubusercontent.com/90977689/156945761-fac20dc1-9391-4b2b-bb34-c4f5e8684d78.png)

![image](https://user-images.githubusercontent.com/90977689/156945782-437dc888-7b4d-4a10-8a09-7cf07c8d9cdc.png)

![image](https://user-images.githubusercontent.com/90977689/156945810-3f2d7333-e780-4ba8-af73-ce0adfa76ecf.png)

![image](https://user-images.githubusercontent.com/90977689/156945891-9c023d5a-42b5-4d61-8489-742740721174.png)









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


## Link to Google Slides Presentation
- [Google Slides](https://docs.google.com/presentation/d/1XoV1bcYuccp9_o8XffJp5emKioTJpqc7fV_RFGwLNw8/edit?usp=sharing)

## Link to Tableau Dashboard
* The tools we used to create our dashboard was searching the web to find related datasets to upload to Tableau.   Once uploaded in Tableau Public, various bar graphs and maps were created to illustrate the overall homeless population and funding received to shelter those individuals.  
* One of the many benefits of using Tableau Public is the program allows you to create interative data visualizations. 
<br>[link to dashboard](https://public.tableau.com/app/profile/michal.upchurch/viz/HICClassProject/TopHomelessPopulationbyCoC?publish=yes "link to dashboard")

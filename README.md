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

The random forest machine learning technique will be used to develop algorithms to classify homelessness as a function of shelter availability, homeless counts and homeless funding. 

## Database
The extracted dataframes from the HUD data will be stored utilizing a SQlite database. Once data is cleaned of irrelevant columns and cleaned to eliminate non-populated data and revise any pieces of data that diverge from the expected value ranges, the tables will be merged based upon the CoC indices as our master key for all values. Because the content of our database is expected to be manageable from a data size standpoint, all files will be saved to the repository, with the ability to read in the data from any project participant individual on their own machine, reducing the need to rely upon a centrally stored database. This will also serve to let each team member manipulate the data on their unique branches to ensure that all content is sercure, yet able to accomodate independent research without burdensome lockout/tag-out procedures to maintain data integrity. The tables we intend to read into the dataframe include: <br>
- CityData
- ShelterAvailability
- HomelessCounts
- HomelessFunding

![Database-Schema](https://user-images.githubusercontent.com/81537476/154897228-6a360a8a-afbf-4d9e-803b-c2e7d086240a.png)


If there is additional need to contextualize the data, we intend to incorporate [metropolitan statistical area data](https://www.census.gov/programs-surveys/metro-micro/about.html) that can assist in providing some demographic and socioeconomic context to the areas supported by the specific CoCs of interest in our analysis.

## Summary

## Comments

## Communication Protocols
- As a group we will be using Slack for a majority of our communication especially during class time and outside of it as well.
- Meeting as a team outside of class hours will take place via Zoom.
- We will be checking out Trello for potential project planning communication.

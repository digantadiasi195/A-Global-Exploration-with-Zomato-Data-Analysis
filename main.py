import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px

def about_dataset(df):
    print(df.head())
    print('\n')
    print(df.info())
    print('\n')
    print(df.describe())
    print('\n')
    print(df.columns)
    missing_info = df.isnull().sum()
    print(missing_info)
    print("\n Missing value columns: \n")
    for features in df.columns:
        if df[features].isnull().sum() > 0:
            print(features)

def make_final_dataFrame(df, df_country_code):
    final_df = pd.merge(df, df_country_code, on='Country Code', how='left')
    print("Combine Data File:\n", final_df)
    print(final_df.columns)
    cnt = final_df.Country.value_counts()
    print(cnt)
    return final_df

def country_chart(final_df):
    country_names = final_df['Country'].value_counts().index
    country_values = final_df['Country'].value_counts().values
    print(country_names)
    print(country_values)
    print(final_df.groupby(['Aggregate rating', 'Country']).size().reset_index().head(5))

    plt.pie(country_values[:3], labels = country_names[:3], autopct='%1.2f%%')
    plt.show()

def observation_Through_Rating(final_df):
    rating_info = final_df.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index().rename(columns = {0:'Rating Count'})
    print(rating_info)
    ''' conclusion:
       when rating is between 4.5 to 4.9 -> Excellent
       when rating is between 4.0 to 3.4 -> Very good
       when rating is between 3.5 to 3.9 -> Good
       when rating is between 3.0 to 3.4 -> Average
       when rating is between 2.0 to 2.4 -> Poor
    '''
    sns.barplot(x="Aggregate rating", y="Rating Count", hue="Rating color", data=rating_info, palette = [ 'white', 'red', 'orange', 'yellow', 'green', 'green'])
    plt.show()

    sns.countplot(x="Rating color", data = rating_info, palette = [ 'white', 'red', 'orange', 'yellow', 'green', 'green'])
    plt.show()
    '''' observation:
      Not rated count is very high
      maximum number of rating are between 2.5 to 3.4
    '''
def country_gives_zero_rating(final_df):
    zero_rating_country = final_df.groupby(['Aggregate rating', 'Country']).size().reset_index().head(5)
    print(zero_rating_country)
   ## Observation: Maximun number of zero rating are from indian customer
    currency_find = final_df.groupby([ 'Country', 'Currency']).size().reset_index().head(10)
    print("Country Used Currency:\n",currency_find)
 

def online_delivery(final_df):
    delivery_online = final_df[['Has Online delivery', 'Country']].groupby(['Has Online delivery', 'Country']).size().reset_index()
    print(delivery_online)
    ## Observatoin: Online delivery are available in India and UAE 

def online_order_VS_rate(final_df):
    plt.figure(figsize=(6, 6))
    sns.boxplot(x='Has Online delivery', y='Aggregate rating', data=final_df)
    plt.title('Impact of Online Order Availability on Aggregate Ratings')
    plt.xlabel('Has Online Delivery (1 for Yes, 0 for No)')
    plt.ylabel('Aggregate Rating')
    plt.show()

    
def city_distribution(final_df):
    city_values = final_df.City.value_counts().values
    city_labels = final_df.City.value_counts().index
    plt.pie(city_values[:5], labels= city_labels[:5], autopct = '%1.2f%%' )
    plt.show()

def correlation_analysis(final_df):
    ## select numerical value column
    numerical_columns = final_df.select_dtypes(include=['float64', 'int64']).columns 
    correlation_matrix = final_df[numerical_columns].corr()

    # Creating a heatmap to visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()    


def cuisine_analysis(final_df):
    # Split the 'Cuisines' column into individual cuisines
    cuisines_split = final_df['Cuisines'].str.split(', ', expand=True).stack()

    # Count the occurrences of each cuisine
    cuisine_counts = cuisines_split.value_counts()

    # Select the top N cuisines (adjust N as needed)
    top_cuisines = cuisine_counts.head(10)

    # Plot a bar chart to visualize the distribution of top cuisines
    plt.figure(figsize=(12, 6))
    top_cuisines.plot(kind='bar', color='skyblue')
    plt.title('Top 10 Cuisines')
    plt.xlabel('Cuisine')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha='right')  
    plt.show()

    # Print the table of top cuisines with their counts
    print("Top 10 Cuisines:\n", top_cuisines)

def price_range_analysis(final_df):
    # Count the occurrences of each price range
    price_range_counts = final_df['Price range'].value_counts()

    # Plot a bar chart to visualize the distribution of price ranges
    plt.figure(figsize=(8, 6))
    price_range_counts.sort_index().plot(kind='bar', color='skyblue')
    plt.title('Distribution of Restaurants by Price Range')
    plt.xlabel('Price Range')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=0)  # Rotate x-axis labels if needed
    plt.show()

    # Print the table of price range counts
    print("Price Range Counts:\n", price_range_counts)


def delivery_options_analysis(final_df):
    # Count the occurrences of each delivery option
    delivery_options_counts = final_df[['Has Online delivery', 'Has Table booking']].apply(pd.value_counts)

    # Plot a bar chart to visualize the counts
    plt.figure(figsize=(10, 6))
    delivery_options_counts.T.plot(kind='bar', stacked=True, color=['skyblue', 'lightgreen'])
    plt.title('Availability of Delivery Options')
    plt.xlabel('Delivery Options')
    plt.ylabel('Number of Restaurants')
    plt.legend(title='Option', loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.show()

    # Print the table of counts
    print("Delivery Options Counts:\n", delivery_options_counts)


def india_analysis(final_df):
    # Filter the DataFrame to include only Indian restaurants
    indian_restaurants = final_df[final_df['Country'] == 'India']

    # Count the occurrences of each city in India
    city_counts = indian_restaurants['City'].value_counts()

    # Plot a bar chart to visualize the distribution of restaurants across Indian cities
    plt.figure(figsize=(14, 6))
    sns.barplot(x=city_counts.index, y=city_counts.values, palette="viridis")
    plt.title('Distribution of Restaurants Across Indian Cities')
    plt.xlabel('City')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.show()

def best_selling_foods(final_df):
    # Group by cuisine and calculate the average rating and total votes
    cuisine_stats = final_df.groupby('Cuisines').agg({'Aggregate rating': 'mean', 'Votes': 'sum'}).reset_index()

    # Sort by average rating or total votes to find the best-selling foods
    best_selling_by_rating = cuisine_stats.sort_values(by='Aggregate rating', ascending=False).head(10)
    best_selling_by_votes = cuisine_stats.sort_values(by='Votes', ascending=False).head(10)

    print("Top 10 Best-Selling Foods by Rating:\n", best_selling_by_rating)
    print("\nTop 10 Best-Selling Foods by Votes:\n", best_selling_by_votes)


def top_ten_cuisines_analysis(final_df):
    # Count the occurrences of each cuisine
    cuisine_counts = final_df['Cuisines'].str.split(', ', expand=True).stack().value_counts()
    top_ten_cuisines = cuisine_counts.head(10)

    # Plot a bar chart to visualize the distribution of the top ten cuisines
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top_ten_cuisines.index, y=top_ten_cuisines.values, palette="viridis")
    plt.title('Top Ten Cuisines')
    plt.xlabel('Cuisine')
    plt.ylabel('Number of Restaurants')
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    plt.show()

def most_popular_cuisines(final_df, top_n=10):
    plt.figure(figsize=(12, 8))
    top_cuisines = final_df['Cuisines'].value_counts().head(top_n)
    sns.barplot(x=top_cuisines.values, y=top_cuisines.index, palette='viridis')
    plt.xlabel('Number of Restaurants')
    plt.ylabel('Cuisines')
    plt.title(f'Top {top_n} Most Popular Cuisines')
    plt.show()



def plot_restaurant_locations_plotly(final_df):
    # Create a scatter map using plotly
    fig = px.scatter_mapbox(final_df, 
                            lat="Latitude", 
                            lon="Longitude", 
                            hover_name="Restaurant Name",
                            zoom=10)

    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html('restaurant_locations_map_plotly.html')


def plot_cost_by_country(df):
    '''
    Plot the average cost for two with respect to each country.
    '''
    avg_cost_by_country = df.groupby('Country')['Average Cost for two'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(14, 8))  
    sns.barplot(x='Country', y='Average Cost for two', data=avg_cost_by_country, palette='viridis')
    plt.title('Average Cost for Two by Country')
    plt.xlabel('Country')
    plt.ylabel('Average Cost for Two')
    plt.xticks(rotation=45, ha='right', fontsize=10)  
    plt.tight_layout()  
    plt.show()

def plot_price_distribution_for_india(df):
    '''
    Plot the price distribution of Indian restaurants.
    '''
    indian_restaurants = df[df['Country'] == 'India']

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(indian_restaurants['Average Cost for two'], bins=20, kde=True, color='skyblue')
    plt.title('Price Distribution of Indian Restaurants')
    plt.xlabel('Average Cost for Two')
    plt.ylabel('Frequency')
    plt.show()


def country_rating_comparison(final_df):
    """
    Compare average ratings by country and identify countries with the highest ratings.

   """
    avg_ratings_by_country = final_df.groupby('Country')['Aggregate rating'].mean().reset_index()
    highest_rated_countries = avg_ratings_by_country.sort_values(by='Aggregate rating', ascending=False).head(3)

    # Plotting
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Country', y='Aggregate rating', data=avg_ratings_by_country, palette='viridis')
    plt.title('Average Ratings by Country')
    plt.xlabel('Country')
    plt.ylabel('Average Rating')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Print summary
    print("Countries with the Highest Average Ratings:")
    print(highest_rated_countries[['Country', 'Aggregate rating']])

    # Check if there is a majority of restaurants with 0 ratings in any country
    zero_rating_counts = final_df[final_df['Aggregate rating'] == 0].groupby('Country').size().reset_index()
    countries_with_majority_zero_ratings = zero_rating_counts[zero_rating_counts[0] > len(final_df) / 2]

    if not countries_with_majority_zero_ratings.empty:
        print("\nCountries with a Majority of Restaurants Having 0 Ratings:")
        print(countries_with_majority_zero_ratings[['Country', 0]])


def main():
    df = pd.read_csv('zomato.csv', encoding='latin-1')
    df_country_code = pd.read_excel('Country_Code.xlsx')
    about_dataset(df)
    final_df = make_final_dataFrame(df, df_country_code)
    country_chart(final_df)
    observation_Through_Rating(final_df)
    country_gives_zero_rating(final_df)
    online_delivery(final_df)
    online_order_VS_rate(final_df)
    city_distribution(final_df)
    correlation_analysis(final_df)
    cuisine_analysis(final_df)
    price_range_analysis(final_df)
    delivery_options_analysis(final_df)
    india_analysis(final_df)
    most_popular_cuisines(final_df, top_n=10)
    plot_cost_by_country(final_df)
    country_rating_comparison(final_df)
    plot_price_distribution_for_india(final_df)
    
    
if __name__ == "__main__":
    main()

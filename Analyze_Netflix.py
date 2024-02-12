# Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Start coding!

# Load the CSV file as netflix_df
netflix_df = pd.read_csv("netflix_data.csv")
print(netflix_df.head())

# Filter the data to remove TV shows as netflix_subset
netflix_subset = netflix_df[netflix_df['type'] == 'Movie']

# Create a DataFrame as netflix_movies
netflix_movies = netflix_subset[["title", "country", "genre", "release_year", "duration"]]

# Filter movies shorter than 60 min as short_movies
short_movies = netflix_movies[netflix_movies["duration"] < 60]
print(short_movies.head())

# Assign colors based on genres
colors = []
for index, row in netflix_movies.iterrows():
    if row["genre"] == "Children":
        colors.append("blue")
    elif row["genre"] == "Documentaries":
        colors.append("green")
    elif row["genre"] == "Stand-Up":
        colors.append("red")
    else:
        colors.append("grey")

# Create Scatter Plot
fig, ax = plt.subplots()
ax.scatter(netflix_movies["release_year"], netflix_movies["duration"], c=colors)

# Set plot labels and title
ax.set_xlabel('Release year')  # Corrected from set_xlabels to set_xlabel
ax.set_ylabel('Duration (min)')  # Corrected from set_ylabels to set_ylabel
ax.set_title('Movie Duration by Year of Release')

# Display the plot
plt.show()
import numpy as np

# Clean and convert the relevant columns
data['sample_size'] = pd.to_numeric(data['sample_size'], errors='coerce')
data['score'] = data['score'].str.replace('%', '').astype(float) / 100  # Convert score to a decimal

# Drop rows with missing or invalid data in these columns
data = data.dropna(subset=['sample_size', 'score'])

# Compute effect size and precision
data['effect_size'] = data['score'] * data['sample_size']
data['standard_error'] = 1 / np.sqrt(data['sample_size'])  # Assuming this relation
data['precision'] = 1 / data['standard_error']

# Display updated dataset
#data[['aspect', 'sample_size', 'score', 'effect_size', 'precision']].head()

# Aggregate effect sizes by broader categories
aggregated_data = data.groupby('broader_category').agg(
    total_effect_size=('effect_size', 'sum'),
    mean_effect_size=('effect_size', 'mean'),
    std_effect_size=('effect_size', 'std'),
    count=('effect_size', 'size')
).reset_index()

# Visualize aggregated effect sizes by broader categories
plt.figure(figsize=(12, 8))
plt.barh(aggregated_data['broader_category'], aggregated_data['total_effect_size'], edgecolor='black', alpha=0.7)
plt.title('Total Effect Size by Broader Categories', fontsize=16)
plt.xlabel('Total Effect Size', fontsize=14)
plt.ylabel('Broader Categories', fontsize=14)
plt.grid(axis='x', alpha=0.75)
plt.show()

# Display the aggregated data for reference
import ace_tools as tools; tools.display_dataframe_to_user(name="Aggregated Effect Sizes by Broader Categories", dataframe=aggregated_data)

# Cochran's Q and I² statistics calculations

# Compute the weighted mean effect size
weighted_mean_effect_size = (
    data['effect_size'] / data['standard_error']**2
).sum() / (1 / data['standard_error']**2).sum()

# Compute Cochran's Q
data['squared_diff'] = (
    ((data['effect_size'] - weighted_mean_effect_size)**2) / data['standard_error']**2
)
cochrans_q = data['squared_diff'].sum()

# Compute degrees of freedom (number of studies - 1)
degrees_of_freedom = len(data) - 1

# Compute I² statistic
i_squared = max(0, (cochrans_q - degrees_of_freedom) / cochrans_q) * 100

# Output results
{
    "Cochran's Q": cochrans_q,
    "Degrees of Freedom": degrees_of_freedom,
    "I² (%)": i_squared
}

# Analyze precision trends over sample sizes
import seaborn as sns
plt.figure(figsize=(10, 6))

# Create a scatter plot to analyze the trend
sns.scatterplot(x=filtered_data['sample_size'], y=filtered_data['precision'])

# Add a line of best fit to visualize trends
sns.regplot(x=filtered_data['sample_size'], y=filtered_data['precision'], scatter=False, color='red', ci=None)

# Add titles and labels
plt.title('Precision Trends Over Sample Sizes', fontsize=16)
plt.xlabel('Sample Size', fontsize=14)
plt.ylabel('Precision', fontsize=14)

plt.grid(True)
plt.show()

# -----------------------------------------------
# extract all LLM/transformers analyzed in each paper
import pandas as pd

# Read the CSV file into a pandas DataFrame
#df = pd.read_csv('data.csv')
#df = pd.read_csv('data.csv', encoding='latin-1')
df = pd.read_csv('data.csv', encoding='iso-8859-1')
#df = pd.read_csv('data.csv', engine='python')

# Group by 'Reference' and aggregate LLM columns as a list (without duplicates)
def combine_llms(group):
    llms = []
    for col in ['LLM1', 'LLM2', 'LLM3']:
        for llm in group[col].dropna().tolist():
            if llm not in llms:  # Add only if not already in the list
                llms.append(llm)
    return llms

refs_df = df.groupby('Reference').apply(combine_llms).reset_index(name='LLMs')

# Export the results to a new CSV file
refs_df.to_csv('refs.csv', index=False)
print("File 'refs.csv' has been created successfully.")

### ==================
import pandas as pd
from collections import Counter
import re
from nltk.corpus import stopwords
from datetime import datetime
import matplotlib.pyplot as plt

try:
    import nltk
    nltk.data.find('corpora/stopwords')
except LookupError:
    import nltk
    nltk.download('stopwords')

# Load the CSV file
try:
    #df = pd.read_csv("data.csv", parse_dates=['date'])
    df = pd.read_csv('data.csv', encoding='iso-8859-1', parse_dates=['date'])
except FileNotFoundError:
    print("Error: data.csv not found. Please make sure the file is in the correct directory.")
    exit()
except ValueError:
      print("Error: could not parse date, check the 'date' column format (e.g., 'YYYY-MM-DD').")
      exit()
except pd.errors.ParserError:
    print("Error: Parsing error in CSV file.  Please check if the file is correctly formatted.")
    exit()

if df.empty:
    print("Error: data.csv is empty.")
    exit()

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str): # Handle missing values or unexpected data types
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text) # Remove punctuation
    text = re.sub(r"\d+", "", text) #remove digits
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return words

# Apply preprocessing
df['processed_text'] = df['text'].apply(preprocess_text)

# Function to calculate word frequencies over time
def analyze_word_trends(df, time_period='year'):

    if time_period not in ['year', 'month']:
         raise ValueError("time_period must be 'year' or 'month'")

    if time_period == 'year':
       df['time_index'] = df['date'].dt.year
       group_key = 'time_index'
    else:
        df['time_index'] = df['date'].dt.to_period('M')
        group_key = 'time_index'


    word_counts_by_period = {}
    for time_index, group in df.groupby(group_key):
        all_words = []
        for text_list in group['processed_text']:
            all_words.extend(text_list)
        word_counts_by_period[time_index] = Counter(all_words)

    # Calculate word frequency trends
    word_trends = {}
    all_words_set = set()

    for counts in word_counts_by_period.values():
      all_words_set.update(counts.keys())

    all_words_list = list(all_words_set)

    for word in all_words_list:
      word_trends[word] = []
      for time_index in sorted(word_counts_by_period.keys()):
         count = word_counts_by_period[time_index].get(word, 0)
         word_trends[word].append((time_index, count))

    return word_trends
#Analyze word trends
try:
    word_trends = analyze_word_trends(df, time_period='year')

    # Identify words with increasing frequency (example)
    threshold = 1 #minimum count to be considered
    increasing_words = {}

    for word, counts_over_time in word_trends.items():
       if len(counts_over_time) < 2: #Need at least 2 time points to check the change
          continue

       increasing = True
       previous_count = counts_over_time[0][1]

       if previous_count < threshold:
           increasing = False

       for time_index, current_count in counts_over_time[1:]:
            if current_count < previous_count and previous_count >= threshold:

               increasing = False
               break

            if previous_count < threshold and current_count < threshold:
                increasing = False
                break

            previous_count = current_count

       if increasing and previous_count >= threshold:
          increasing_words[word] = counts_over_time


    # Print or visualize results
    if increasing_words:
        print("Words with increasing frequency over time:")

        for word, counts_over_time in increasing_words.items():
           print(f"\nWord: {word}")
           for time_index, count in counts_over_time:
              print(f"  {time_index}: {count}")

           time_points = [str(x[0]) for x in counts_over_time]
           counts = [x[1] for x in counts_over_time]

           plt.figure(figsize=(10, 5))
           plt.plot(time_points, counts, marker='o')
           plt.xlabel("Time Period (Year)")
           plt.ylabel("Frequency")
           plt.title(f"Frequency of '{word}' Over Time")
           plt.grid(True)
           plt.tight_layout()
           plt.show()


    else:
        print("No words found with consistently increasing frequency over time (with the current settings). Try decreasing the threshold.")
except ValueError as e:
     print(f"ValueError: {e}")
except Exception as e:
     print(f"An unexpected error occurred: {e}")


#example for month level analysis

try:
    word_trends_monthly = analyze_word_trends(df, time_period='month')

    # Identify words with increasing frequency (example)
    threshold = 1 #minimum count to be considered
    increasing_words_monthly = {}

    for word, counts_over_time in word_trends_monthly.items():
       if len(counts_over_time) < 2: #Need at least 2 time points to check the change
          continue

       increasing = True
       previous_count = counts_over_time[0][1]

       if previous_count < threshold:
           increasing = False

       for time_index, current_count in counts_over_time[1:]:
            if current_count < previous_count and previous_count >= threshold:

               increasing = False
               break

            if previous_count < threshold and current_count < threshold:
                increasing = False
                break

            previous_count = current_count

       if increasing and previous_count >= threshold:
          increasing_words_monthly[word] = counts_over_time


    # Print or visualize results
    if increasing_words_monthly:
        print("\n\nWords with increasing frequency over time (monthly):")

        for word, counts_over_time in increasing_words_monthly.items():
           print(f"\nWord: {word}")
           for time_index, count in counts_over_time:
              print(f"  {time_index}: {count}")

           time_points = [str(x[0]) for x in counts_over_time]
           counts = [x[1] for x in counts_over_time]

           plt.figure(figsize=(10, 5))
           plt.plot(time_points, counts, marker='o')
           plt.xlabel("Time Period (Month)")
           plt.ylabel("Frequency")
           plt.title(f"Frequency of '{word}' Over Time (monthly)")
           plt.grid(True)
           plt.tight_layout()
           plt.show()


    else:
        print("No words found with consistently increasing frequency over time (monthly analysis, with the current settings). Try decreasing the threshold or choosing a longer period for analysis.")
except ValueError as e:
     print(f"ValueError: {e}")
except Exception as e:
     print(f"An unexpected error occurred: {e}")

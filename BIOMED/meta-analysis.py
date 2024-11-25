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

### === Visualizing ranges of scores
# Convert the Score column to numeric (if needed)
data['Score'] = pd.to_numeric(data['Score'], errors='coerce')

# Group by 'Keyword' and calculate min and max scores for each keyword
score_ranges = data.groupby('Keyword')['Score'].agg(['min', 'max']).reset_index()

# Create a bar plot to visualize the ranges
plt.figure(figsize=(10, 6))
for idx, row in score_ranges.iterrows():
    plt.plot([row['min'], row['max']], [idx, idx], marker='o', label=row['Keyword'] if idx == 0 else "")

plt.yticks(range(len(score_ranges)), score_ranges['Keyword'])
plt.xlabel('Score')
plt.title('Score Ranges for Each Keyword')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

### == visualizing conditions as circles colored by scores and sized by occurences
import matplotlib.pyplot as plt
import numpy as np
import csv

def visualize_circles(csv_file):
    """
    Visualizes data from a CSV file as circles with sizes proportional to scores
    and colors assigned based on a gradient between predefined colors,
    interpolated according to the 'Color' column value (0-100).

    Args:
        csv_file (str): Path to the CSV file.
    """

    conditions = []
    scores = []
    color_values = []  # Store raw color values from the CSV (0-100)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            conditions.append(row['Conditions'])
            try:
                scores.append(int(row['Score']))
            except ValueError:
                print(f"Warning: Invalid score '{row['Score']}' found for condition '{row['Conditions']}'. Skipping this entry.")
                continue
            try:
                color_values.append(float(row['Color']))  # Store the raw color value (0-100)
            except ValueError:
                print(f"Warning: Invalid color value '{row['Color']}' found for condition '{row['Conditions']}'. Skipping this entry.")
                continue

    # Define 20 colors from blue to yellow to match VOSViewer
    color_list = [    
    (0.00, 0.00, 0.55),  # Dark Blue (Viridis 0.0)
    (0.06, 0.11, 0.55),  # Dark Blue (Viridis 0.05)
    (0.13, 0.23, 0.55),  # Dark Blue-Green (Viridis 0.1)
    (0.19, 0.33, 0.53),  # Dark Blue-Green (Viridis 0.15)
    (0.26, 0.42, 0.46),  # Dark Green-Blue (Viridis 0.2)
    (0.16, 0.49, 0.56),  # Teal Blue 25%
    (0.25, 0.09, 0.51),  # Deep Blue
    (0.20, 0.15, 0.60),  # Dark Blueish
    (0.15, 0.28, 0.63),  # Blue
    (0.35, 0.71, 0.65),  # Soft Green
    (0.42, 0.81, 0.36),  # Mantis 75%
    (0.58, 0.67, 0.84),  # AK  
    (0.64, 0.83, 0.11),  # Dark Yellow-Green (Viridis 0.5)
    (0.85, 0.95, 0.00),  # Dark Golden (Viridis 0.7)
    (0.93, 0.98, 0.00),  # Dark Amber (Viridis 0.8)
    (0.96, 0.99, 0.00),  # Dark Amber (Viridis 0.85)
    (0.98, 0.99, 0.00),  # Dark Orange-Yellow (Viridis 0.9)
    (1.00, 1.00, 0.00),  # Strong Yellow (Viridis 1.0)
]


    # Create the plot
    plt.figure(figsize=(10, 10))
    plt.title("Conditions Visualized by Circle Size and Color")

    # Calculate positions for diagonal arrangement
    num_circles = len(scores)
    x = np.linspace(0.1, 0.9, num_circles)  # From left to right
    y = 1 - x  # Diagonal from top-left to bottom-right

    for i, (score, color_value, condition) in enumerate(zip(scores, color_values, conditions)):
        # Scale the score for circle size
        circle_size = score * 10  # Adjust this multiplier to fit your data range

        # Determine color based on color_value
        color_index = int(color_value / (100 / (len(color_list) - 1)))
        color_index = min(color_index, len(color_list) - 1)
        color = color_list[color_index]

        # Plot the circle at the calculated position with the assigned color
        plt.scatter(x[i], y[i], s=circle_size, c=[color], edgecolor='black', label=condition)

        # Add label for the condition
        plt.annotate(condition, (x[i], y[i]), xytext=(0, 10), textcoords='offset points', ha='center', va='bottom')

    # Ensure circles are not distorted
    plt.axis('equal')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_circles('conditions.csv')

### === Mapping to dermatology keywords
keywords_list = [
    "Viral infection", "Infestation", "Benign tumors", "Eczema", "Fungal infections", "Alopecia", "Acne", 
    "Psoriasis", "Bacterial infection", "Pigmentary disorders", "Alopecia areata", "Androgenetic alopecia", 
    "Central centrifugal cicatricial alopecia", "Traction alopecia", "Actinic Keratosis", "Acne vulgaris", 
    "Atopic dermatitis", "Bullous pemphigoid", "Epidermolysis bullosa", "Herpes zoster", "Lamellar ichthyosis", 
    "Lichen planus", "Poroma", "Ganglionic cyst", "Ecchymosis", "Ecthyma", 
    "Infantile Hemangioma", "Nutritional dermatoses", "injury", "ulcers", "Nonmelanoma skin cancer", 
    "Inflammatory skin conditions", "Autoimmune diseases", "Pigmented lesions", 
    "Cutaneous lymphoma", "Skin infections", "Rosacea", "Hidradenitis suppurativa", 
    "Desmoplastic Trichoepithelioma", "DPTE", "Extramammary Paget Disease", "EMPD", "Leiomyosarcoma", "Melanoma", "CTCL", 
    "cutaneous T-cell lymphoma", "Merkel Cell Carcinoma", "MCC", "Squamous Cell Carcinoma", "SCC", "Angiosarcoma", "Basal Cell Carcinoma", "BCC", 
    "Lentigo Maligna", "LM", "Lentigo Maligna Melanoma", "LMM", "Atypical Fibroxanthoma", "AFX", "Bowenoid Papules", 
    "Dermatofibrosarcoma Protuberans", "DFSP", "Sebaceous Carcinoma", "Cutaneous tumors", "Dermatitis", 
    "Dyspigmentation", "Shingles", "Fungal nail infections", "Pityriasis versicolor", "Tinea corporis", 
    "Impetigo", "Erythrasma", "Dermatomyositis", "Scleroderma", "Morphea", "Discoid lupus erythematosus", 
    "Clavus", "corn", "Warts", "Pyoderma gangrenosum", "Abscess", "Atopic eczema", "Chronic urticaria", 
    "Skin cancer", "Non-melanoma", "Papulosquamous", "Papule", "Hair Shedding", "Benign Lesion", "skin disease"
]

# Function to map the text to the keywords
def map_keywords(text, keywords_list):
    matches = [kw for kw in keywords_list if kw.lower() in text.lower()]
    top_keyword = matches[0] if matches else "N/A"
    other_keywords = "; ".join(matches[1:]) if len(matches) > 1 else "N/A"
    return top_keyword, other_keywords

# Apply the function to map the data with updated keywords
data[['Top Keyword', 'Other Keywords']] = data['Text'].apply(
    lambda x: pd.Series(map_keywords(str(x), keywords_updated))
)

# Export the final mapped data
output_path = 'mapped.csv'
data.to_csv(output_path, index=False)

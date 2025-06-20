import pandas as pd

# Load the dataset
df = pd.read_csv("DimensionsExport.csv")

# Clean header
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

# Define target models and comparative keywords
models = ["grok 3", "gpt o3", "gemini 2.5"]
comparative_keywords = ["compared to", "versus", "vs.", "benchmark", "evaluation", 
                        "outperformed", "performed better than", "compared with"]

# Function to check if abstract mentions a target model and a comparative keyword
def has_comparative_eval(text, model_keywords, comparison_keywords):
    if not isinstance(text, str):
        return False
    lower_text = text.lower()
    return any(model in lower_text for model in model_keywords) and \
           any(kw in lower_text for kw in comparison_keywords)

# Apply filtering
df["Comparative_Model"] = df["Abstract"].apply(
    lambda x: next((model for model in models if isinstance(x, str) and model in x.lower()), None)
)
mask = df.apply(lambda row: has_comparative_eval(row["Abstract"], models, comparative_keywords), axis=1)
comparative_df = df[mask]

# Save or display
comparative_df.to_csv("Filtered_Comparative_Evaluations.csv", index=False)

# If using in an interactive environment:
#import ace_tools as tools
#tools.display_datafra_

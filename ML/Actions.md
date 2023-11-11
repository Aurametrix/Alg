https://platform.openai.com/docs/actions


# Define the GPT that will handle the user input and output
gpt = GPT(name="Website Browser", description="A GPT that can browse a specific website and find a specific file")

# Define the action that will perform the browsing and finding task
action = Action(name="Browse and Find", description="An action that can browse a specific website and find a specific file")

# Define the input parameters for the action
action.add_param(name="website", type="string", description="The website URL to browse")
action.add_param(name="file", type="string", description="The file name to find")

# Define the output parameters for the action
action.add_output(name="result", type="string", description="The result of the browsing and finding task")

# Define the logic for the action
action.set_logic("""
# Import the requests library to make HTTP requests
import requests

# Get the website URL from the input parameter
website = params.get("website")

# Get the file name from the input parameter
file = params.get("file")

# Make a GET request to the website and store the response
response = requests.get(website)

# Check if the response status code is 200 (OK)
if response.status_code == 200:
    # Get the content of the response as text
    content = response.text
    
    # Check if the file name is in the content
    if file in content:
        # Set the output parameter result to a success message
        output.set("result", f"Found the file {file} on the website {website}")
    else:
        # Set the output parameter result to a failure message
        output.set("result", f"Could not find the file {file} on the website {website}")
else:
    # Set the output parameter result to an error message
    output.set("result", f"Could not access the website {website}")
""")

# Add the action to the GPT
gpt.add_action(action)

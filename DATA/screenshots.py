# https://chromedriver.chromium.org/downloads
# pip install selenium Pillow

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import os

def take_snapshot(url, output_filename):
    # Set up Chrome options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--window-size=1920,1080")  # Set the window size
    
    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)
    
    # Navigate to the URL
    driver.get(url)
    
    # Take a screenshot
    driver.save_screenshot("full_screenshot.png")
    
    # Get the size of the screenshot
    img = Image.open("full_screenshot.png")
    width, height = img.size
    
    # Calculate the bottom right quadrant
    # Assuming the quadrant is defined as the lower right quarter of the screen
    left = width / 2
    top = height / 2
    right = width
    bottom = height
    
    # Crop the image
    img_cropped = img.crop((left, top, right, bottom))
    
    # Save the cropped image
    img_cropped.save(output_filename)
    
    # Clean up
    driver.quit()
    os.remove("full_screenshot.png")  # Remove the temporary full screenshot

# Example usage
url = "https://www.example.com"
output_filename = "bottom_right_quadrant.png"
take_snapshot(url, output_filename)

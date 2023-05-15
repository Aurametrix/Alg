###
* [BioGPT](https://github.com/microsoft/BioGPT) - the implementation of [2022 article](https://academic.oup.com/bib/article/23/6/bbac409/6713511?guestAccessKey=a66d9b5d-4f83-4017-bb52-405815c907b9&login=false)
* [BlenderGPT](https://github.com/gd3kr/BlenderGPT) - Generate Blender Python code from natural language commands
* [ViperGPT](https://viper.cs.columbia.edu/) - Visual Inference via Python Execution for Reasoning
* [ChatGPT talks to your AWS infrastructure footprint](https://www.akitasoftware.com/blog-posts/we-built-an-exceedingly-polite-ai-dog-that-answers-questions-about-your-apis)
* [ChatGPR & Whisper APIs](https://openai.com/blog/introducing-chatgpt-and-whisper-apis)
* [List of alternatives to ChatGPT](https://github.com/nichtdax/awesome-totally-open-chatgpt)
* [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html)
* [QuiLLMan](https://github.com/modal-labs/quillman) - Voice Chat with LLMs
* [WEB LLM](https://github.com/mlc-ai/web-llm) - [bringing chatbots to web browsers](https://mlc.ai/web-llm/)
* [IPython-GPT](https://github.com/santiagobasulto/ipython-gpt) - IPython ChatGPT extension
* [chat-gpt-jupyter-extension](https://github.com/jflam/chat-gpt-jupyter-extension) 
* [jupytee](https://github.com/fperez/jupytee) 
* [jupyter-voicepilot](https://github.com/JovanVeljanoski/jupyter-voicepilot)
* [Prompt Engineering](https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf)
* [ChatHUB](https://github.com/chathub-dev/chathub/blob/main/README.md) - Use ChatGPT, Bing, Bard and Claude in One App 




### LLAMA
+ [FB LLAMA](https://github.com/facebookresearch/llama)
+ [Finetune LLaMA-7B on commodity GPUs using your own text](https://github.com/lxe/simple-llama-finetuner)
+ [Minimal LLAMA](https://github.com/zphang/minimal-llama/)) - code for running and fine-tuning LLaMA.
+ [Low-Rank LLaMA Instruct-Tuning](https://github.com/lxe/simple-llama-finetuner ); [py](https://github.com/tloen/alpaca-lora/blob/main/finetune.py)
+ [Run LLAMA and ALPACA on your computer](https://github.com/cocktailpeanut/dalai)

### Programmable ChatGPT
+ [ChatGPT wrapper](https://github.com/mmabrouk/chatgpt-wrapper)
+ [Prompt injection on Bing Chat](https://greshake.github.io/)
+ [Jailbreak Chat](https://www.jailbreakchat.com/)
+ [WordGPT](https://github.com/filippofinke/WordGPT)

+ [Visual ChatGPT](https://github.com/microsoft/visual-chatgpt) - connects ChatGPT and a series of Visual Foundation Models to enable sending and receiving images during chatting
+       [AI tokenizer](https://platform.openai.com/tokenizer)
+       [Tiktoken](https://github.com/openai/tiktoken)
+       [Tiktokenizer](https://tiktokenizer.vercel.app/)

+ [Plugins](https://openai.com/blog/chatgpt-plugins)
    + [wolfram](https://writings.stephenwolfram.com/2023/03/chatgpt-gets-its-wolfram-superpowers/)

###
[Chatbot kit](https://chatbotkit.com/)


The following models are considered GPT 3.5:

- code-davinci-002

- text-davinci-002

- text-davinci-003


###
[jupyter-extension](https://github.com/TiesdeKok/chat-gpt-jupyter-extension) - A browser extension to provide various helper functions in Jupyter Notebooks and Jupyter Lab, powered by ChatGPT.
[social](https://github.com/riverscuomo/social) - A python package that uses OpenAI to generate a response to a social media mention

[Freely Available GPT models](https://huggingface.co/EleutherAI/gpt-neo-2.7B)


[reverse engineered chatGPT API](https://github.com/acheong08/ChatGPT)

[built with GPT](https://github.com/elyase/awesome-gpt3)

[Basic Search for a Website](https://platform.openai.com/docs/tutorials/web-qa-embeddings/how-to-build-an-ai-that-can-answer-questions-about-your-website): 
[full code](https://github.com/openai/openai-cookbook/tree/main/solutions/web_crawl_Q%26A)

[Open Assistant](https://github.com/LAION-AI/Open-Assistant) - with LAION AI - [main page](https://open-assistant.io/)

[GPTalk 0.0.4.4](https://github.com/0ut0flin3/GPTalk)

[github support bot with GPT3](https://dagster.io/blog/chatgpt-langchain)

###
https://github.com/mmabrouk/chatgpt-wrapper
pip install git+https://github.com/mmabrouk/chatgpt-wrapper
(uses playwright: https://playwright.dev/)
pip freeze | grep playwright 
playwright==1.28.0
chatgpt install
The SDK provides a command-line interface

### The second open-source SDK is chatgpt-python
https://github.com/labteral/chatgpt-python


### Official instructions

Python Flask
git clone https://github.com/openai/openai-quickstart-python.git
If you prefer not to use git, you can alternatively download the code using this zip file.

Add your API key
Navigate into the project directory and make a copy of the example environment variables file.

cd openai-quickstart-python
cp .env.example .env
Copy your secret API key and set it as the OPENAI_API_KEY in your newly created .env file. If you haven't created a secret key yet, you can do so below.

SECRET KEY	CREATED	LAST USED	
sk-...S3Tk
Dec 5, 2022	Never	

Run the app
Run the following commands in the project directory to install the dependencies and run the app. When running the commands, you may need to type python3/pip3 instead of python/pip depending on your setup.

python -m venv venv
. venv/bin/activate
pip install -r requirements.txt
flask run
Open http://localhost:5000 in your browser and you should see the pet name generator!

Understand the code
Open up app.py in the openai-quickstart-python folder. At the bottom, you’ll see the function that generates the prompt that we were using above. Since users will be entering the type of animal their pet is, it dynamically swaps out the part of the prompt that specifies the animal.

def generate_prompt(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(animal.capitalize())
On line 14 in app.py, you’ll see the code that sends the actual API request. As mentioned above, it uses the completions endpoint with a temperature of 0.6.


response = openai.Completion.create(
  model="text-davinci-003",
  prompt=generate_prompt(animal),
  temperature=0.6
)

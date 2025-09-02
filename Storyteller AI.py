# %% [markdown]
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# %% [markdown]
# # <a id='toc1_'></a>[Use Mistral and gTTS to Create Your Personal Storyteller](#toc0_)
# 

# %% [markdown]
# Estimated time needed: **30** minutes
# 
# 
# In this project, you will learn how to use Mistral and gTTS to create your personal storyteller.
# 

# %% [markdown]
# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#Introduction">Introduction</a></li>
#     <li><a href="#What-does-this-guided-project-do?">What does this guided project do?</a></li>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Background">Background</a>
#         <ol>
#             <li><a href="#What-is-large-language-model-(LLM)?">What is large language model (LLM)?</a></li>
#             <li><a href="#What-is-Mistral?">What is Mistral?</a></li>
#             <li><a href="#What-is-gTTS-(Google-Text-to-Speech)?">What is gTTS (Google Text-to-Speech)?</a></li>
#         </ol>
#     </li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
#         </ol>
#     </li>
#     <li><a href="#watsonx-API-credentials-and-project_id">watsonx API credentials and project_id</a></li>
#     <li>
#         <a href="#Work-with-foundation-models-on-watsonx.ai">Work with foundation models on watsonx.ai</a>
#         <ol>
#             <li><a href="#List-available-models">List available models</a></li>
#             <li><a href="#Defining-model-parameters">Defining model parameters</a></li>
#         </ol>
#     </li>
#     <li><a href="#Generate-a-story-with-Mistral">Generate a story with Mistral</a></li>
#     <li><a href="#Convert-the-story-to-speech">Convert the story to speech</a></li>
#     <li><a href="#Save-the-audio-to-a-file">(Optional) Save the audio to a file</a></li>
#     <li>
#         <a href="#Exercises">Exercises</a>
#         <ol>
#             <li><a href="#Exercise-1:-Generate-another-story">Exercise 1: Generate another story</a></li>
#         </ol>
#     </li>
#     <li><a href="#Authors">Authors</a></li>
#     <li><a href="#Contributors">Contributors</a></li>
# </ol>
# 

# %% [markdown]
# <h2 id="Introduction"><a href="#Table-of-Contents">Introduction</a></h2>
# 
# Have you ever wanted to create engaging stories and have them read aloud naturally? By combining the power of AI story generation with text-to-speech technology, we can create an interactive storytelling experience. In this project, we'll use Mistral, a large language model, to generate creative stories based on any topic you provide, and then convert these stories into natural-sounding speech.
# 
# <h2 id="What-does-this-guided-project-do"><a href="#Table-of-Contents">What does this guided project do?</a></h2>
# 
# 
# This project demonstrates how to create an AI storyteller by:
# 1. Using Mistral to generate creative and informative stories based on your chosen topic
# 2. Converting the generated story into speech using gTTS (Google Text-to-Speech)
# 3. Playing the audio directly in your Jupyter notebook
# 
# For example, you could input a topic like "the life span of trees," and Mistral will create an engaging narrative about how trees grow, survive through seasons, and can live for hundreds or even thousands of years. This story will then be converted into spoken words, making it perfect for educational content, bedtime stories, or learning about any subject in an auditory format.
# 
# <h2 id="Objectives"><a href="#Table-of-Contents">Objectives</a></h2>
# 
# 
# After completing this lab you will be able to:
# - Use Mistral to generate creative stories from any given topic
# - Convert the generated text to speech using the gTTS library
# - Create an end-to-end pipeline for AI storytelling
# - Play generated audio directly in Jupyter notebooks
# 

# %% [markdown]
# <h2 id="Background"><a href="#Table-of-Contents">Background</a></h2>
# 
# <h3 id="What-is-large-language-model-(LLM)?"><a href="#table-of-contents">What is large language model (LLM)?</a></h3>
# 
# 
# [Large language models](https://www.ibm.com/topics/large-language-models?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Use+Mixtral+and+gTTS+to+create+your+personal+storyteller-v1_1738273977) are a category of foundation models trained on immense amounts of data making them capable of understanding and generating natural language and other types of content to perform a wide range of tasks.
# 
# <h3 id="What-is-Mistral"><a href="#Table-of-Contents">What is Mistral?</a></h3>
# 
# [Mistral](https://mistral.ai/) is an open-source large language model developed by Mistral AI. It's a Mixture of Experts (MoE) model that achieves state-of-the-art performance among open-source models. Key features include:
# 
# - **Powerful Performance**: Matches or exceeds the performance of much larger models on most benchmarks
# - **Efficient Architecture**: Uses a Sparse Mixture of Experts architecture, making it more efficient than traditional models
# - **Versatile Applications**: Excellent at tasks like creative writing, analysis, and storytelling
# - **Open Source**: Freely available for research and commercial use
# 
# <h3 id="What-is-gTTS"><a href="#Table-of-Contents">What is gTTS (Google Text-to-Speech)?</a></h3>
# 
# 
# [gTTS (Google Text-to-Speech)](https://gtts.readthedocs.io/) is a Python library and CLI tool that interfaces with Google Translate's text-to-speech API. It offers:
# 
# - **Multiple Languages**: Support for a wide variety of languages and accents
# - **Natural Sound**: High-quality, natural-sounding voice synthesis
# - **Easy Integration**: Simple Python interface for converting text to speech
# - **Format Options**: Ability to save audio in MP3 format or stream it directly
# - **Customization**: Control over speech speed and language variants
# 

# %% [markdown]
# <h2 id="Setup"><a href="#Table-of-Contents">Setup</a></h2>
# 
# For this lab, we will be using the following libraries:
# 
# *   [`ibm-watsonx-ai`](https://pypi.org/project/ibm-watsonx-ai/): `ibm-watsonx-ai` is a library that allows to work with watsonx.ai service on IBM Cloud and IBM Cloud for Data. Train, test and deploy your models as APIs for application development, share with colleagues using this python library.
# *   [`gtts`](https://pypi.org/project/gtts/): `gtts` is a library that allows to convert text to speech using Google Text-to-Speech API.
# 

# %% [markdown]
# <h3 id="Installing-required-libraries"><a href="#Table-of-Contents">Installing required libraries</a></h3>
# 
# 
# The following required libraries are __not__ pre-installed in the Skills Network Labs environment. __You must run the following cell__ to install them. Please wait until it completes.
# 
# This step could take **several minutes**, please be patient.
# 
# **NOTE**: If you encounter any issues, please restart the kernel and run again.  You can do that by clicking the **Restart the kernel** icon.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/crvBKBOkg9aBzXZiwGEXbw/Restarting-the-Kernel.png" width="100%" alt="Restart kernel">
# 

# %%
#%pip install gTTS==2.5.4 | tail -n 1
#%pip install ibm-watsonx-ai==1.1.20 | tail -n 1

# %% [markdown]
# <h2 id="watsonx-API-credentials-and-project_id"><a href="#Table-of-Contents">watsonx API credentials and project_id</a></h2>
# 
# 
# 
# This section provides you with the necessary credentials to access the watsonx API.
# 
# **Please note:**
# 
# In this lab environment, you don't need to specify the api_key, and the project_id is pre_set as "skills-network", but if you want to use the model locally, you need to go to [watsonx](https://www.ibm.com/watsonx?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-Use+Mixtral+and+gTTS+to+create+your+personal+storyteller-v1_1738273977) to create your own keys and id.
# 

# %%
from ibm_watsonx_ai import Credentials
import os

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    )

project_id="skills-network"

# %% [markdown]
# <h2 id="Work-with-foundation-models-on-watsonx.ai"><a href="#Table-of-Contents">Work with foundation models on watsonx.ai</a></h2>
# 
# 
# <h3 id="List-available-models"><a href="#Table-of-Contents">List available models</a></h3>
# 
# 

# %%
from ibm_watsonx_ai import APIClient

client = APIClient(credentials)
# GET TextModels ENUM
client.foundation_models.TextModels

# PRINT dict of Enums
client.foundation_models.TextModels.show()

# %%
# Specify the model_id of the model we will use for the chat.

model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'

# %% [markdown]
# <h3 id="Defining-model-parameters"><a href="#Table-of-Contents">Defining model parameters</a></h3>
# 

# %%
import os
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams


params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 1000,
}

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

# %% [markdown]
# <h2 id="Generate-a-story-with-Mistral"><a href="#Table-of-Contents">Generate a story with Mistral</a></h2>
# 
# Now we'll create a story using Mistral. We'll first define a function that takes a topic as input and returns a generated story. The function will use a carefully crafted prompt to ensure the story is engaging, educational, and appropriate for beginners.
# 
# Let's test our storytelling capabilities by generating a story about a simple topic and converting it to speech.
# 

# %%
# Function to generate an educational story using the Mistral model
def generate_story(topic):
    # Construct a detailed prompt that guides the model to:
    # - Write for beginners
    # - Use simple language
    # - Include interesting facts
    # - Keep a specific length
    # - End with a summary
    prompt = f"""Write an engaging and educational story about {topic} for beginners. 
            Use simple and clear language to explain basic concepts. 
            Include interesting facts and keep it friendly and encouraging. 
            The story should be around 200-300 words and end with a brief summary of what we learned. 
            Make it perfect for someone just starting to learn about this topic."""
    
    # Generate text using the model with our carefully crafted prompt
    response = model.generate_text(prompt=prompt)
    return response

# Example usage of the generate_story function
# Here we use butterflies as a topic since it's an engaging and 
# educational subject that demonstrates the function well
topic = "the life cycle of butterflies"
story = generate_story(topic)
print("Generated Story:\n", story)

# %% [markdown]
# <h2 id="Convert-the-story-to-speech"><a href="#Table-of-Contents">Convert the story to speech</a></h2>
# 
# 
# Now that we have generated our story, let's convert it to speech using the gTTS (Google Text-to-Speech) library.
# We'll create an audio file in memory and play it directly in the notebook using an audio player widget.
# 
# This step may take a while to complete, please be patient.
# 
# **NOTE**: If you encounter any issues, please run the cell again.
# 

# %%
from gtts import gTTS
from IPython.display import Audio
import io

# Initialize text-to-speech with the generated story
tts = gTTS(story)

# Save the audio to a bytes buffer in memory
audio_bytes = io.BytesIO()
tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)

# Create and display an audio player widget in the notebook
Audio(audio_bytes.read(), autoplay=False)

# %% [markdown]
# <h2 id="Save-the-audio-to-a-file"><a href="#Table-of-Contents">(Optional) Save the audio to a file</a></h2>
# 
# 
# ```python
# # Save as MP3 file
# tts.save("generated_story.mp3")
# ```
# 

# %% [markdown]
# <h2 id="Exercises"><a href="#Table-of-Contents">Exercises</a></h2>
# 
# <h3 id="Exercise-1:-Generate-another-story"><a href="#Table-of-Contents">Exercise 1: Generate another story</a></h3>
# 
# 
# Please generate another story with the following topic.
# 
# topic = "the life cycle of a human"
# 
# 

# %%
human_topic = "the life cycle of a human"
human_story = generate_story(human_topic)
print("Generated Story:\n", human_story)

human_tts = gTTS(human_story)

# Save the audio to a bytes buffer in memory
audio_bytes = io.BytesIO()
human_tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)

# Create and display an audio player widget in the notebook
Audio(audio_bytes.read(), autoplay=False)

# %% [markdown]
# <details>
#     <summary>Click here for Solution
#     </summary>
# 
# ```python
# topic = "the life cycle of a human"
# story = generate_story(topic)
# print("Generated Story:\n", story)
# 
# # Initialize text-to-speech with the generated story
# tts = gTTS(story)
# 
# audio_bytes = io.BytesIO()
# tts.write_to_fp(audio_bytes)
# audio_bytes.seek(0)
# 
# # Create and display an audio player widget in the notebook
# Audio(audio_bytes.read(), autoplay=True)
# ```
# </details>
# 

# %% [markdown]
# <h2 id="Authors"><a href="#Table-of-Contents">Authors</a></h2>
# 
# 
# [Ricky Shi](https://author.skills.network/instructors/ricky_shi)
# 
# <h2 id="Contributors"><a href="#Table-of-Contents">Contributors</a></h2>
# 
# [Hailey Quach](https://www.haileyq.com/)
# 
# [Joseph Santarcangelo](https://author.skills.network/instructors/joseph_santarcangelo)
# 
# ```{|Date (YYYY-MM-DD)|Version|Changed By|Change Description||-|-|-|-||2025-01-30|1.0|Ricky Shi|Create project|}
# ```
# 
# 
# 
# 

# %% [markdown]
# Copyright © IBM Corporation. All rights reserved.
# 



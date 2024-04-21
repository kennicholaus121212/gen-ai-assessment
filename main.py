from flask import Flask, render_template, request
import os
import yaml
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel

app = Flask(__name__)

# Helper function that reads from the config file. 
def get_config_value(config, section, key, default=None):
    """
    Retrieve a configuration value from a section with an optional default value.
    """
    try:
        return config[section][key]
    except:
        return default

# Open the config file (config.yaml)
with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Read application variables from the config fle
TITLE = get_config_value(config, 'app', 'title', 'Ask Google')
SUBTITLE = get_config_value(config, 'app', 'subtitle', 'Your friendly Bot')
CONTEXT = get_config_value(config, 'palm', 'context',
                           'You are a bot who can answer all sorts of questions')
BOTNAME = get_config_value(config, 'palm', 'botname', 'Google')
TEMPERATURE = get_config_value(config, 'palm', 'temperature', 0.8)
MAX_OUTPUT_TOKENS = get_config_value(config, 'palm', 'max_output_tokens', 256)
TOP_P = get_config_value(config, 'palm', 'top_p', 0.8)
TOP_K = get_config_value(config, 'palm', 'top_k', 40)
PROJECT_ID = get_config_value(config, 'palm', 'PROJECT_ID', 'qwiklabs-gcp-03-987c5e88d2d4')
API_KEY = get_config_value(config, 'palm', 'API_KEY', 'AIzaSyAyVy43qaTt6ICVRJYdlWEQ0qz7Oltjhnk')



from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=LOCATION)


# The Home page route
@app.route("/", methods=['POST', 'GET'])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == 'GET':
        question = ""
        answer = "Hi, How Can I Help You to clarify questions on restaurant safety?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else: 
        question = request.form['input']

        # Get the data to answer the question that 
        # most likely matches the question based on the embeddings
        data = search_vector_database(question)

        # Ask Gemini to answer the question using the data 
        # from the database
        answer = ask_gemini(question, data)
        
    # Display the home page with the required variables set
    model = {"title": TITLE, "subtitle": SUBTITLE,
             "botname": BOTNAME, "message": answer, "input": question}
    return render_template('index.html', model=model)


def search_vector_database(question):

    def text_embedding(text_to_embed) -> list:
    """Text embedding with a Large Language Model."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@002")
    embeddings = model.get_embeddings([text_to_embed])
    for embedding in embeddings:
        vector = embedding.values
        print(f"Length of Embedding Vector: {len(vector)}")
    return vector

    # emb1 = text_embedding(question)

    # 1. Convert the question into an embedding
    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    # 3. Get the IDs for the five embeddings that are returned
    # 4. Get the five documents from Firestore that match the IDs
    # 5. Concatenate the documents into a single string and return it

    data = ""
    return data


def ask_gemini(question, data):
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    response = "Not implemented!"
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

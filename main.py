from flask import Flask, render_template, request
import os
import yaml
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import Content, GenerativeModel, Part

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
PROJECT_ID = 'qwiklabs-gcp-03-987c5e88d2d4'
API_KEY = 'AIzaSyAyVy43qaTt6ICVRJYdlWEQ0qz7Oltjhnk'
LOCATION = 'us-central1'


from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=LOCATION)


# from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("fpc-manual.pdf")
# pages = loader.load_and_split()
# print(pages[0])

# import json
# FILE_NAME = "embeddings.jsonl"
# # create a jsonl file 
# with open(FILE_NAME, 'w') as f:
#     for id, embedding in zip(ids, embeddings_array):
#         f.writer(json.dumps({"id":str(id), "embedding": embedding.tolist()}) + )


# #create a firestore database for pages
# from google.cloud import firestore

# db = firestore.Client()

# #create a collection for search
# collection_name = 'pdf_pages'
# if collection_name not in collections():
#     db.collection(collection_name)

# #add documents in the collection
# for id, page in zip(ids, pages):
#     db.collection(collection_name).document(str(id)).set(("page":page))

# #query the firestore databaase based on document id 10
# document = db.collection(collection_name).document(str(10)).get()
# print(document.to_dict()["page"][:200])


# #create a cloud storage bucket if it doesnt exist
# from google.cloud import storage
# BUCKET = 'gen_ai_app_ken'
# BUCKET_URI = "gs://{}".format(BUCKET_NAME)

# storage_client = storage.Client()
# bucket = storage_client.bucket(BUCKET_NAME)
# if not bucket.exists():
#     bucket.create(location='us-central1')

# #upload the JSON L file to the bucket
# blob = bucket.blob(FILE_NAME)
# blob.upload_from_filename(FILE_NAME)

# print(BUCKET_URI)


# create vector search index

# create index
#  my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
#     display_name = "gen-ai-index",
#     contents_delta_uri = BUCKET_URI,
#     dimensions = 768,
#     approximate_neighbors_count = 5,
#  )

#  # create index endpoint

# my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
#     display_name = f"gen-ai-index-endpoint"
#     public_endpoint_enabled = True
# )

# # deploy the index
# my_index_endpoint.deploy_index(
#     index = my_index, deployed_index_id = "gen-ai-index-deployed"
# )


#query the vector database
#question =
#QUESTION=

# question_with_task_type = TextEmbeddingInput(
#     text=QUESTION,
#     task_type='RETRIVAL_QUERY'
# )

# from vertexai.language_models import  TextEmbeddingInput

# my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
#     index_endpoint_name ="project/..."
# )

# QUESION_EMBEDDING = textembedding(question_with_task_type)

# respone = my_index_endpoint.find_neighbors(
#     deployed_index_id ="gen-ai-index-endpoint"
#     queries =[QUESION_EMBEDDING],
#     num_neighbors = 5
# )


# #show the search results

# for idex, neighbor in enumerate(response(0)):
#     print(f"{neighbor.distance:.2f} {neighbor.id}")


# # get the pages from firestore

# documents = {}
# for idex, neighbor in enumerate(respone(0)):
#     id = str(neighbor.id)
#     document = db.collection(collection_name).document(id).get()
#     documents.append(document.to_dict["page"])

# pages = "\n\n".join(documents)
# print(len(pages))


# import vertexai
# from vertexai.preview.generative.models import GenerativeModel, Part

# def generate(prompt):
#     model = GenerativeModel("gemini pro")
#     response = model.generate_content(
#         prompt,
#         generation_config=(
#             "max_output_tokens":8192,
#             "temperature":0.5,
#             "top_p":0.5,
#             "top_k":10,
#         ),
#     stream=False,
#     )
#     return response.text

# prompt = '''
# context: edit the following data surrounded by triple back ticks.
# 1. Correct spelling and grammer mistakes.
# 2. Remove data not related to restaurant and food safety
# 3. return the edited data.
# Data: {0}
# cleaned data:
# '''.formate(pages)
# cleaned_data = generate(prompt)


# prompt = '''
# context: Answer the question using the following data surrounded by triple back ticks.
# data: {0}
# question: {1}
# answer:
# '''.format(cleaned_data, QUESTION)








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

    emb1 = text_embedding(question)

    # 1. Convert the question into an embedding
    # 2. Search the Vector database for the 5 closest embeddings to the user's question
    # 3. Get the IDs for the five embeddings that are returned
    # 4. Get the five documents from Firestore that match the IDs
    # 5. Concatenate the documents into a single string and return it

    data = emb1
    return data


def ask_gemini(question, data):
    model = GenerativeModel("gemini-1.0-pro")
    chat = model.start_chat()

    response = chat.send_message(question)
    # You will need to change the code below to ask Gemni to
    # answer the user's question based on the data retrieved
    # from their search
    #response = "Not implemented!"
    return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

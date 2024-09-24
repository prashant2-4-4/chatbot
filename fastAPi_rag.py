from PyPDF2 import PdfReader
from elasticsearch import Elasticsearch
import logging
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr
import io

app = FastAPI(root_path = "/xlscout-invalidator-rerank",
    version = "0.0.1",
    docs_url = '/docs',
    redoc_url = '/redoc',
    openapi_url = '/openapi.json',
    swagger_ui_init_oauth={
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": 'xlscout-main-dashboard'
        })

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

es = Elasticsearch("http://localhost:9200")


from pymongo import MongoClient
m_client  = MongoClient("mongodb://127.0.0.1:28018")

db = m_client["Invalidator_rerank"]

data_base = db["database_questions"]


# ind = es.cat.indices(format = 'json')


reader = PdfReader("iesc111.pdf")
logging.warning(f"Total Pages:  {len(reader.pages)}")

total_text = ""
for i in range(len(reader.pages)):
    page = reader.pages[i]
    # print(page.extract_text())
    total_text += page.extract_text()

print(len(total_text))

total_text_split = total_text.split()
# total_text_split[2]

logging.warning(f"Length of word {len(total_text_split)}")

def chunk_fun(data , split_len = 300):
    chunk_lst = []
    for i in range(0 , len(data) , 300):
        chunk_lst.append(" ".join(data[i:i+300]))
    return chunk_lst


chunk_lst = chunk_fun(total_text_split)


settings = {
    "analysis": {
        "analyzer": {
            "custom_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "english_stemmer"]
            }
        },
        "filter": {
            "english_stemmer": {
                "type": "stemmer",
                "language": "english"
            }
        }
    }
}
mappings = {
    "properties": {
            "vector": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True},
            "text":{ "type":"text",
            "analyzer": "custom_analyzer"}
    }
}


my_index = "ncert"
if not es.indices.exists(index = my_index):
    es.indices.create(index =   my_index , settings=settings , mappings = mappings)


def get_embeddings(single_data):
    return model.encode([single_data])[0]


for i , text in enumerate(chunk_lst):
    es.index(index = my_index , id = i ,body= {"text" : text , "vector" : get_embeddings(text)})


@app.get("/")
def read_root():
    return {"Ping" : "Pong"}


class data_format(BaseModel):
    user_query: str


@app.post("/user_query_first")
def user_query_first(query_user : data_format):
    user_query = query_user.user_query
    query_vector = get_embeddings(user_query)

    cosine_similaity_query = {
    "query" : {
        "script_score" : {"query" : 
         {"match_all" : {} 
         } , 
        "script": { "source" : """


cosineSimilarity(params.query_vector , 'vector') + 1.0
        """ ,
        "params" : {
            "query_vector" : query_vector
        }
        }
        }
    } 
}
    try:
        response = es.search(index = my_index , body=cosine_similaity_query )
    except Exceptions as e:
        logging.warning(f"Exceptions : {e}")

    data_base.insert_one({"query_questions" : user_query , "answer" : response['hits']['hits'][0]['_source']['text']})

    return response['hits']['hits'][0]['_source']['text']
    # for hits in response['hits']['hits']:
    #     print(hits['_score'] , hits['_source']['text'])
    #     break

@app.post("/user_query_second")
def user_query_second(query_user : data_format):
    user_query = query_user.user_query
    query_vector = get_embeddings(user_query)

    cosine_similaity_query = {
    "query" : {
        "script_score" : {"query" : 
         {"match_all" : {} 
         } , 
        "script": { "source" : """


cosineSimilarity(params.query_vector , 'vector') + 1.0
        """ ,
        "params" : {
            "query_vector" : query_vector
        }
        }
        }
    } 
}
    #We can add more case here...
    basic_questions_list = {'hello' : "hello how can i help you" , 'hi' : 'hi how can i help you', 'how are you' : "good how can i assist you today."}

    #implementing basic questions logic
    for k , v in basic_questions_list.items():
        k_embed = get_embeddings(k)
        if(cosine_similarity([k_embed] ,[query_vector]) > 0.85):
            return basic_questions_list[k]


    #check if the user already asked the questions before.
    existing = data_base.find_one({"query_questions" : user_query})
    if existing:
        return existing['answer']
    else:
        try:
            response = es.search(index = my_index , body=cosine_similaity_query )
        except Exceptions as e:
            logging.warning(f"Exceptions : {e}")

        data_base.insert_one({"query_questions" : user_query , "answer" : response['hits']['hits'][0]['_source']['text']})

        return response['hits']['hits'][0]['_source']['text']



@app.post("/voice-input/")
async def voice_input(file: UploadFile = File(...)):
    logging.warning("here")
    contents = await file.read()
    audio_file = io.BytesIO(contents)

    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Convert speech to text
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            logging.warning(f"Text received is {text}")
            query = data_format(user_query=text)
            out = user_query_second(query)
            return {"recognized_text": text, "response": out}
        except sr.UnknownValueError:
            return {"error": "Speech recognition could not understand audio"}
        except sr.RequestError as e:
            return {"error": f"Could not request results from speech recognition service; {e}"}
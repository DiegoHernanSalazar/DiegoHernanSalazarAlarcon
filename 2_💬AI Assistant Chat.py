# STEP 1
import streamlit as st # Utilized for constructing the 
                       # user interface of the chatbot, 
                       # enabling user interactions 
                       # and visual display of the chat interface.
                       
import torch # This library is integrated to leverage GPU 
             # capabilities, ensuring the efficient processing 
             # and computation, especially beneficial when 
             # handling deep learning models.

from llama_index import (GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, 
                         ServiceContext, LangchainEmbedding)
                         # This is employed as the data access 
                         # layer for AI, facilitating advanced 
                         # data retrieval and query functionalities.

from langchain.embeddings import HuggingFaceInstructEmbeddings 
                                 # This is used for embedding 
                                 # purposes, which can involve 
                                 # converting words or phrases 
                                 # into numerical vectors, 
                                 # thereby facilitating the 
                                 # efficient processing and 
                                 # handling of textual data 
                                 # within machine learning models.

# ibm_watson_machine_learning library allows seamless interaction 
# with the Watsonx.ai service, enabling the application 
# to leverage IBM Watson’s machine learning capabilities.
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

from utils.constants import * # All constants from 'utils' library
                              # are imported.

# STEP 2
# -- Use Streamlit 'st' constructor to create a simple user interface -- #
st.title("💬 Chat with My AI Assistant") # Display input 'text'
                                          # in title formatting.
def local_css(file_name): # Create function to open, read an apply 
                          # the input file_name.css, to the Streamlit
                          # app with customized appearance, via
    # st.markdown('string of HMTL <tag>s'.format(f.read()), allow_html=True)

    with open(file_name) as f: # open input HTML file with <tag>s
        
        # Add HTML <tag>s opened input file 'f', to 'HTML input text'
        # and say any HTML <tag> found at input opened 
        # string file 'f', not being treated as pure text.
        # It will be treated as HTML format with <tag>s.
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_chat.css") # Call the 'local_css()' function
                                  # created above, including as input
                                  # parameter the string with
                                  # "folder/input file name.css" 
                                  # on cloned GitHub.

# STEP 3
# Let’s call the variables stored in utils/constants.py
# Get the variables or 'key' values from 'constants.py', 
# at 'info' dictionary, as: value = info['Key'].
pronoun = info['Pronoun']     # "his"
name = info['Name']           # "Diego" 
subject = info['Subject']     # "he"
full_name = info['Full_Name'] # "Diego Hernán Salazár Alarcón"

# Initialize the chat history
if "messages" not in st.session_state: # If there isn't exist
                                       # "messages" =___.messages = [] list, 
                                       # then create that list with initial
                                       # assistant message [ {} ].

    # Initial assistant´s message value, is created
    # to be updated at 'content' key
    welcome_msg = f"Hi! I'm {name}'s AI Assistant, Pelufo. How may I assist you today?"
    
    # Fill the list of "messages"=___.messages = [], with initial message 
    # from assistant to user [ {1st message from assistant to user} ]
    st.session_state.messages = [{"role": "assistant", "content": welcome_msg}]


# Add App sidebar using 'with' notation
with st.sidebar:

    # Add a 'Markdown' as text (not as HTML), 
    # keeping unsafe_allow_html = False, as default.
    st.markdown("""
                # Chat with my AI assistant 
                """)
    
    # Add a multi-element container that can be expanded or collapsed,
    # using 'with' notation.
    with st.expander("Click here to see FAQs"):

        # Informational 'text' to display
        st.info(
            f"""
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest lab or project?
            - When can {subject} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        ) # Close info 'text' to display
    
    import json # Library to convert list of [{messages}] into a JSON/Dict file.

    messages = st.session_state.messages # List of messages [ {} ]
    if messages is not None: # If list of messages [ {} ] is not empty []

        # Display a download button widget, to download a file directly from app.    
        st.download_button(
            label="Download Chat",     # Button label
            data=json.dumps(messages), # 'data' file to be downloaded as JSON/Dict
            file_name='chat.json',     # Optional "string" to use as the name of file
                                       # to be downloaded
            mime='json',               # Type of the data downloaded. If None,
                                       # defaults to 'text/plain'
        ) # Close Download button widget
    
    # Display text in small font, used for captipons, asides, footnotes, sidenotes
    # and other explanatory text.
    st.caption(f"© Made by {full_name} 2024. All rights reserved.")


# STEP 4
# Replace the placeholder "Watsonx_API" and "Project_id" with your 
# actual 'API key' and 'Project ID' to configure Watsonx API.

# Temporarily displays a message while executing a block of code. 
# Use 'with' notation.
with st.spinner("Initiating the AI assistant. Please hold..."):
    # CODE TO BE EXECUTED    
    # Check for GPU availability and set the appropriate device for computation.
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Global variables
    llm_hub = None
    embeddings = None
    
    # Copy your own API Key
    Watsonx_API = "BoqPnwZ4ZZKjh0JWwwSnO-sNdeEq33MsddIIyEEVQfb-"

    # Copy your own Project ID 
    Project_id = "26e30d40-c76c-49f2-98f5-d6ef0dfe52e7"   

    # Function to initialize the language model and its embeddings
    def init_llm():
        global llm_hub, embeddings # Define global variables
        
        # Tokens are smaller units that can be processed by the LLM AI models
        # Tokens can be words, characters, subwords, or symbols
        # Tokenization means splitting the input and output texts into smaller units
        params = {
            GenParams.MAX_NEW_TOKENS: 512, # The maximum number of tokens that the model can generate in a single run.
            GenParams.MIN_NEW_TOKENS: 1,   # The minimum number of tokens that the model should generate in a single run.
            GenParams.DECODING_METHOD: DecodingMethods.SAMPLE, # The method used by the model for decoding/generating new tokens. In this case, it uses the sampling method.
            GenParams.TEMPERATURE: 0.7,   # A parameter that controls the randomness of the token generation. A lower value (close to 0) makes the generation more deterministic,
                                          # while a higher value (close to 1) introduces more randomness, exploration and variety in the model's responses. 
            GenParams.TOP_K: 50,          # The top K parameter restricts the token generation to the K most likely tokens at each step, which can help to focus the generation
                                          # and avoid irrelevant tokens. 
            GenParams.TOP_P: 1            # The top P parameter, also known as nucleus sampling, restricts the token generation to a subset of tokens that have a cumulative
                                          # probability of at most P, helping to balance between diversity and quality of the generated text.
        }
        
        # Dictionary including IBM 'url' and own 'apikey' as credentials of access
        credentials = {
            'url': "https://us-south.ml.cloud.ibm.com",
            'apikey' : Watsonx_API
        }
        
        # model_name = LLAMA_2_70B_CHAT
        model_id = ModelTypes.LLAMA_2_70B_CHAT
        
        # Use 'Model()' class, that includes model_name, credentials dict,
        # parameters dict and own Project_id, as input variables, then
        # assign this to model=Model(model_name, credentials, parameters, Project_id)
        # object.
        model = Model(
            model_id= model_id,
            credentials=credentials,
            params=params,
            project_id=Project_id) # Close 'Model()' class
        
        # Use imported 'WatsonxLLM()' class, with input parameter, the model
        # (object) constructor of type 'Model()' class. That is stored at 
        # 'llm_hub' global variable, which was initiated as empty/None.
        llm_hub = WatsonxLLM(model=model)
    
        # Initialize embeddings for text transformation tasks,
        # using a pre-trained model to represent the transformed text data.
        # Text transformation tasks are language translation, spelling and grammar checking, 
        # tone adjustment, and format conversion.
        embeddings = HuggingFaceInstructEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
        )
    
    init_llm() # Call previously created function, out of 'def init_llm():'
               # but inside of 'with'.
    
    # Load the biography 'bio.txt' file with 'llama_index' library, 'SimpleDirectoryReader()' class,
    # and '.load_data()' function/method.
    documents = SimpleDirectoryReader(input_files=["bio.txt"]).load_data()
    
    # From 'llama_index' library, 'LLMPredictor()' class is used
    # to generate the text response (Output Completion)
    llm_predictor = LLMPredictor(
            llm=llm_hub
    )
                                    
    # Hugging Face text transformation models, can be supported by using 
    # 'LangchainEmbedding()' class, from 'llama_index' library,
    # to convert text to embedding vector. (Numerical representation of
    # text (data), that captures semantic (meaning or interpretation) 
    # and similarities between words, symbols, expressions 
    # and formal representations of all linguistic signs).	
    embed_model = LangchainEmbedding(embeddings)
    
    # From 'llama_index' library, use 'ServiceContext()' class, 
    # 'from_defaults()' function/method, to encapsulate/cluster  
    # the resources used, to create indexes and run queries.    
    service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor, # 1st resource used is output completion (response)
            embed_model=embed_model      # 2nd resource used is Numerical representation of Transformed 'text'
    )
      
    # Build index from input document (Create an overview of data, through indexing 
    # documents), with 'llama_index' library, 'GPTVectorStoreIndex()' class,  
    # and 'from_documents()' function/method, using 'bio.txt' document as input, 
    # WatsonxLLM(Model()) output completion (model response) and 'embed_model' numerical representation of 'embeddings'
    # transformed 'text', as 2 input resources clustered/encapsulated at 'service_context'.
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)


# STEP 5
# Build a Query Engine with 'llama_index' library, by summiting the instructions
# from 'system' to AI 'assistant', including the AI assistant behaviour.

def ask_bot(user_query):

    global index   # Define 'index' as global variable in 'ask_bot' function.
    
    # Give the Context from 'system' role to AI 'assistant' role as a 
    # prompt/string value = PROMPT_QUESTION of 'content' key.
    # Remember that any previous 'text' variable, goes as {variable}, 
    # to be included inside any prompt.

    PROMPT_QUESTION = """You are Pelufo, an AI assistant dedicated to assisting {name} in {pronoun} job search by providing recruiters with relevant information about
    {pronoun} qualifications and achievements.  
    Your goal is to support {name} in presenting {pronoun} self effectively to potential employers and promoting {pronoun} candidacy for job opportunities.
    If you do not know the answer, politely admit it and let recruiters know how to contact {name} to get more information directly from {pronoun}. 
    Don't put "Pelufo" or a breakline in the front of your answer.
    Human: {input}
    """
    
    # Query 'llama_Index' library and 'LLAMA_2_70B_CHAT' model to get the AI's response.
    # Use 'as_query_engine()' constructor, that takes a Natural Language query/prompt/
    # input 'text' string, and returns a rich response output, usually built on one
    # or many indexes, vía retrievers. Use '.query()' function to include the 
    # input / prompt / context of 'text', called PROMPT_QUESTION in string format.
    # Use "prompt with text".format(name=name, pronoun=pronoun, input=user_query) to pass
    # 'text' variables taken from 'constants.py' at STEP 3.
    
    output = index.as_query_engine().query(PROMPT_QUESTION.format(name=name, pronoun=pronoun, input=user_query)) # Get the model output completion (object), from input prompt
                                                                                                                 # which includes context from 'system' to 'assistant',
                                                                                                                 # and the 'user_query' = 'input' question from 'user'.
    return output                                                                                                # Return the model's output completion (object)


# STEP 6

# Wait for/Accept a user's input 'prompt'/message, and append that message to the message history list
# [ {message} ], at the value of key 'content'. if prompt := 'user's input message' uses := operator 
# at if statement, which means 'prompt' variable is defined to be whatever value 'user's input message' is.
if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

# Iterate through the messages history list [{msg1}, {msg2}, ..., {msgn}] 
# and display each separated {message}.
for message in st.session_state.messages:  # Pass each individual message
    
    # Insert a Chat message container per 'role' ('user' or 'assistant' value)
    # using 'with' notation, and then include the 'content' value.
    with st.chat_message(message["role"]): # Set msg container as 'user' or 'assistant'
         st.write(message["content"])      # Then include user (prompt) or assistant (response) values 
    

# If the last 'role' message is not from the 'assistant', 
# then generate a new assistant/model (response obj).response
if st.session_state.messages[-1]["role"] != "assistant": # If 'role' key value is not 'assistant',
                                                         # so  'user' != 'assistant'
    with st.chat_message("assistant"):     # Set msg container at 'assistant', using 'with' notation
        with st.spinner("🤔 Thinking..."): # Temporarily displays ':) Thinking...' message, 
                                           # while executes the next code, using 'with' notation
            response = ask_bot(prompt)     # model output (response obj), from last user input prompt
            st.write(response.response)    # write output.response from model, at msg container
            message = {"role": "assistant", "content": response.response} # Create model/assistant {message} to be appended
            st.session_state.messages.append(message) # Append model response {message} to messages list/history 
                                                      # [...,{msgn-1},{msgn}]


# STEP 7
# User choices are buttons that a chatbot displays to a user. 
# The chatbot visitor would click and choose one of the many choices the bot 
# displays. For example, user choices can be used to display the most frequently 
# asked questions for a user to choose from.

# Suggested list with questions
questions = [
    f'What are {pronoun} strengths in AI, Machine Learning, Data Science, Robotics, Control and Software Development?',
    f'What are some of {pronoun} main projects as part of {pronoun} AI training?',
    f'What is {subject} doing or running currently as Generative AI enthusiast?'
]

# Create 'send_button_question()' function, with a user's 'question' 
# as input of the full prompt at STEP 5. 
def send_button_ques(question):        # question = user_query = input
    st.session_state.disabled = True   # When any question button is clicked, the disabled() 
                                       # function is executed, and that disables 
                                       # the button (1st thing that occurs).

    response = ask_bot(question)       # Then, get the model's output (response obj), from the user's 'question' input
    st.session_state.messages.append({"role": "user", "content": question}) # Append to 'messages' list, 
                                                                            # the 'user' question [...,{msgn-1}, {msgn}]
    st.session_state.messages.append({"role": "assistant", "content": response.response}) # and append to 'messages' list 
                                                                                          # the AI 'assistant' response
                                                                                          # [...,{msgn-1}, {msgn}]

# Out of 'send_button_question()' function    
if 'button_question' not in st.session_state: # If 'st.session_state' dictionary, 
                                              # hasn't the {'button_question':""} key and value included,
    st.session_state['button_question'] = ""  # then create it, with an empty string value ""

if 'disabled' not in st.session_state:    # If 'st.session_state' dictionary, 
                                          # hasn't the {'disabled':False} key and value included,
    st.session_state['disabled'] = False  # then create it, with a 'False' boolean as value, to
                                          # keep any button enabled to be clicked on.
    
if st.session_state['disabled'] == False: # When 'disabled' key was recently created with 'False' boolean value
                                          # because it didn't exist at dictionary,
    
    for n, msg in enumerate(st.session_state.messages): # Pass ALL 'messages' {msgn} 
                                                        # from the list [{msg1}, {msg2},...,{msgn}],
                                                        # enumerating each {msg} as n[0 -> N-1] integer.
        # Render suggested question buttons
        # Inserts an invisible container into your app that can be used to hold multiple elements. 
        # This allows you to, for example, insert multiple elements into your app out of order.
        # To add elements to the returned container, you can use "with" notation (preferred) 
        # or just call methods directly on the returned object. 
        buttons = st.container()  # Insert a multi-element container that will keep 
                                  # 3 suggested questions on it, from the beginning, 
                                  # at n=0 -> {msg0}. 'buttons' is also a streamlit (object).
                                  
        if n == 0:                # When {msg0} with n=0, is iterated/submmited,
            for q in questions:   # pass the list with user's suggested, 3 questions.
                
                # Using uniquely created container out of inner for loop (previously),
                # pass each user's suggested question 'q' to display 'st.button()' function, as 'label' input parameter
                # Use 'st.button()' to display a single button with 'label', or 'buttons_streamlit_obj.button()' to display multiple buttons.
                # We don´t use 'st.container().button()' inside questions for loop, because we'd create 3 different containers, each with 1 msg each iter.
                # Otherwise, we need one or the same container (streamlit_object), to be filled with 3 'streamlit_obj.button()', 
                # including one different question/label input each iter.
                # Input parameters are: button label/question, function executed when clicked, function input argument/question, button disabled value is 'False'
                # for keeping it enabled, before being clicked on.
                button_ques = buttons.button(label=q, on_click=send_button_ques, args=[q], disabled=st.session_state.disabled) # Add a button widget to
                                                                                                                               # 'buttons' container, each iter.

import streamlit as st
import pandas as pd
import os
import joblib
from openai import OpenAI
from pinecone import Pinecone
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from streamlit_chat import message

if "btn_clicked" not in st.session_state:
        st.session_state.btn_clicked = False

def callback():
    st.session_state.btn_clicked = True

load_dotenv()

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),)

# Connect to your Pinecone index
pinecone_index = pc.Index("stocks")

original_data_file_path = './data/original_data.csv'

original_data = pd.read_csv(original_data_file_path)

original_data = original_data[['VIX','BDIY', 'LUMSTRUU', 'LUACTRUU', 'LF94TRUU','XAU BGNL', 'CRY', 'DXY', 'JPY', 'GBP','MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXCN','USGG30YR','USGG3M', 'EONIA']]

scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(original_data)

def load_all_models_from_folder(folder_path):
    """
    Load all .pkl models from the specified folder.

    Args:
        folder_path (str): Path to the folder containing .pkl model files.

    Returns:
        dict: A dictionary where keys are model filenames (without .pkl extension) and values are the loaded models.
    """
    models = {}
    try:
        # Iterate through all files in the folder
        for file in os.listdir(folder_path):
            if file.endswith('.pkl'):  # Check for .pkl files
                model_name = os.path.splitext(file)[0]  # Get the filename without extension
                model_path = os.path.join(folder_path, file)
                models[model_name] = joblib.load(model_path)  # Load the model
        return models
    except Exception as e:
        print(f"Error loading models: {e}")
        return {}


def perform_rag(query, model_output):
    raw_query_embedding = get_huggingface_embeddings(query)

    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(),
        top_k=5,
        include_metadata=True,
        namespace="https://github.com/CoderAgent/SecureAgent"
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[:10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as needed to improve the response quality
    system_prompt = f"""You are a seasoned financial expert with over 20 years of experience in the stock market. You possess in-depth knowledge of all aspects of stocks and investments, and your advice has consistently turned ordinary investors into millionaires. Your expertise lies in analyzing complex data and translating it into actionable, professional-level investment strategies.

    Given a detailed analysis of the stock market (JSON object: {model_output}), your task is to provide clear, concise, and actionable investment guidance.

    Never reference or mention the JSON object.
    Use the data exclusively to craft meaningful insights and strategies for the user.
    Respond as a professional financial advisor, ensuring your answer is direct, precise, and actionable.
    If the user provides a query, answer it with thoughtful analysis based on your expertise.
    If the user's query is vague or empty, analyze the provided data comprehensively and present the best investment strategies tailored to the current market conditions.
    Your advice should always be presented in a professional tone, focusing on maximizing the user's financial growth.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content
def voting(arr):
    count = arr.count(1)
    count0 = arr.count(0)

    return 1 if (count >= count0) else 0



def make_prediction(normalised_user_input):
    iso_forest_predictions1 = st.session_state.models['isolation_forest_model_Currency and Commodities Gold'].predict(normalised_user_input[['XAU BGNL', 'CRY', 'DXY', 'JPY', 'GBP']])
    log_reg_prediction1 = st.session_state.models['logistic_regression_model_Currency and Commodities Gold'].predict(normalised_user_input[['XAU BGNL', 'CRY', 'DXY', 'JPY', 'GBP']])
    mlp_prediction1 = st.session_state.models['neural_network_model_Currency and Commodities Gold'].predict(normalised_user_input[['XAU BGNL', 'CRY', 'DXY', 'JPY', 'GBP']])

    iso_forest_predictions1_binary = (iso_forest_predictions1 == -1).astype(int)

    s1_pred = voting([iso_forest_predictions1_binary[0], log_reg_prediction1[0], mlp_prediction1[0]])

    s1 = {
        'Strategy': "Currency and Commodities Gold",
        'Strategy_prediction' : s1_pred,
        'Isolation Forest' : iso_forest_predictions1_binary[0],
        'Logistic Regression' : log_reg_prediction1[0],
        'Neural Network' : mlp_prediction1[0],
        'indexes': 'XAU BGNL, CRY, DXY, JPY, GBP'
    }

    iso_forest_predictions2 = st.session_state.models['isolation_forest_model_Economic Sentiment'].predict(normalised_user_input[['USGG30YR','USGG3M', 'EONIA', 'LUMSTRUU', 'LUACTRUU']])
    log_reg_prediction2 = st.session_state.models['logistic_regression_model_Economic Sentiment'].predict(normalised_user_input[['USGG30YR','USGG3M', 'EONIA', 'LUMSTRUU', 'LUACTRUU']])
    mlp_prediction2 = st.session_state.models['neural_network_model_Economic Sentiment'].predict(normalised_user_input[['USGG30YR','USGG3M', 'EONIA', 'LUMSTRUU', 'LUACTRUU']])

    iso_forest_predictions2_binary = (iso_forest_predictions2 == -1).astype(int)

    s2_pred = voting([iso_forest_predictions2_binary[0], log_reg_prediction2[0], mlp_prediction2[0]])

    s2 = {
        'Strategy': "Economic Sentiment and Rates Interest Rates",
        'Strategy_prediction' : s2_pred,
        'Isolation Forest' : iso_forest_predictions2_binary[0],
        'Logistic Regression' : log_reg_prediction2[0],
        'Neural Network' : mlp_prediction2[0],
        'indexes': 'USGG30YR,USGG3M, EONIA, LUMSTRUU, LUACTRUU'
    }


    iso_forest_predictions3 = st.session_state.models['isolation_forest_model_Equity Market Indicators'].predict(normalised_user_input[['MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXCN']])
    log_reg_prediction3 = st.session_state.models['logistic_regression_model_Equity Market Indicators'].predict(normalised_user_input[['MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXCN']])
    mlp_prediction3 = st.session_state.models['neural_network_model_Equity Market Indicators'].predict(normalised_user_input[['MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXCN']])

    iso_forest_predictions3_binary = (iso_forest_predictions3 == -1).astype(int)

    s3_pred = voting([iso_forest_predictions3_binary[0], log_reg_prediction3[0], mlp_prediction3[0]])

    s3 = {
        'Strategy': "Equity Market Indicators Regional Market Indices",
        'Strategy_prediction' : s3_pred,
        'Isolation Forest' : iso_forest_predictions3_binary[0],
        'Logistic Regression' : log_reg_prediction3[0],
        'Neural Network' : mlp_prediction3[0],
        'indexes': 'MXUS, MXEU, MXJP, MXBR, MXCN'
    }

    iso_forest_predictions4 = st.session_state.models['isolation_forest_model_Liquidity and Money Market Indicators'].predict(normalised_user_input[['LUMSTRUU', 'LUACTRUU', 'LF94TRUU']])
    log_reg_prediction4 = st.session_state.models['logistic_regression_model_Liquidity and Money Market Indicators'].predict(normalised_user_input[['LUMSTRUU', 'LUACTRUU', 'LF94TRUU']])
    mlp_prediction4 = st.session_state.models['neural_network_model_Liquidity and Money Market Indicators'].predict(normalised_user_input[['LUMSTRUU', 'LUACTRUU', 'LF94TRUU']])

    iso_forest_predictions4_binary = (iso_forest_predictions4 == -1).astype(int)

    s4_pred = voting([iso_forest_predictions4_binary[0], log_reg_prediction4[0], mlp_prediction4[0]])

    s4 = {
        'Strategy': "Liquidity and Money Market Indicators",
        'Strategy_prediction' : s4_pred,
        'Isolation Forest' : iso_forest_predictions4_binary[0],
        'Logistic Regression' : log_reg_prediction4[0],
        'Neural Network' : mlp_prediction4[0],
        'indexes': 'LUMSTRUU, LUACTRUU, LF94TRUU'
    }

    iso_forest_predictions5 = st.session_state.models['isolation_forest_model_Volatility Indicators'].predict(normalised_user_input[['VIX', 'BDIY']])
    log_reg_prediction5 = st.session_state.models['logistic_regression_model_Volatility Indicators'].predict(normalised_user_input[['VIX', 'BDIY']])
    mlp_prediction5 = st.session_state.models['neural_network_model_Volatility Indicators'].predict(normalised_user_input[['VIX', 'BDIY']])

    iso_forest_predictions5_binary = (iso_forest_predictions5 == -1).astype(int)

    s5_pred = voting([iso_forest_predictions5_binary[0], log_reg_prediction5[0], mlp_prediction5[0]])

    s5 = {
        'Strategy': "Volatility Indicators",
        'Strategy_prediction' : s5_pred,
        'Isolation Forest' : iso_forest_predictions5_binary[0],
        'Logistic Regression' : log_reg_prediction5[0],
        'Neural Network' : mlp_prediction5[0],
        'indexes': 'VIX, BDIY',
    }

    return {'overall_pred' : voting([s1['Strategy_prediction'],s2['Strategy_prediction'],s3['Strategy_prediction'],s4['Strategy_prediction'],s5['Strategy_prediction']]),
            "Strategies": [
                s1,s2,s3,s4,s5
            ]
            }

    

st.session_state.models = load_all_models_from_folder('./models/')
print("model loaded")

# Initialize default values in session state if not already set
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        'LUMSTRUU': 870.940,
        'LUACTRUU': 474.045,
        'LF94TRUU': 990.750,
        'XAU BGNL': 283.250,
        'CRY': 157.260,
        'DXY': 100.560,
        'JPY': 105.860,
        'GBP': 1.646,
        'MXUS': 1416.120,
        'MXEU': 127.750,
        'MXJP': 990.590,
        'MXBR': 856.760,
        'MXCN': 34.300,
        'USGG30YR': 6.671,
        'USGG3M': 5.426,
        'EONIA': 2.890,
        'VIX': 22.500,  
        'BDIY': 1388.000 
    }

# Function to update session state inputs
def update_input(key, value):
    st.session_state.inputs[key] = value

# App title
st.title("Market Crash Prediction App")

# Sidebar instructions
st.sidebar.markdown("Adjust the values below to simulate different scenarios.")

# Group inputs by categories
st.sidebar.header("Liquidity and Money Market Indicators")
st.session_state.inputs['LUMSTRUU'] = st.sidebar.number_input(
    "LUMSTRUU", value=st.session_state.inputs['LUMSTRUU'], on_change=update_input, args=("LUMSTRUU", st.session_state.inputs['LUMSTRUU'])
)
st.session_state.inputs['LUACTRUU'] = st.sidebar.number_input(
    "LUACTRUU", value=st.session_state.inputs['LUACTRUU'], on_change=update_input, args=("LUACTRUU", st.session_state.inputs['LUACTRUU'])
)
st.session_state.inputs['LF94TRUU'] = st.sidebar.number_input(
    "LF94TRUU", value=st.session_state.inputs['LF94TRUU'], on_change=update_input, args=("LF94TRUU", st.session_state.inputs['LF94TRUU'])
)

st.sidebar.header("Currency and Commodities Gold")
st.session_state.inputs['XAU BGNL'] = st.sidebar.number_input(
    "XAU BGNL", value=st.session_state.inputs['XAU BGNL'], on_change=update_input, args=("XAU BGNL", st.session_state.inputs['XAU BGNL'])
)
st.session_state.inputs['CRY'] = st.sidebar.number_input(
    "CRY", value=st.session_state.inputs['CRY'], on_change=update_input, args=("CRY", st.session_state.inputs['CRY'])
)
st.session_state.inputs['DXY'] = st.sidebar.number_input(
    "DXY", value=st.session_state.inputs['DXY'], on_change=update_input, args=("DXY", st.session_state.inputs['DXY'])
)
st.session_state.inputs['JPY'] = st.sidebar.number_input(
    "JPY", value=st.session_state.inputs['JPY'], on_change=update_input, args=("JPY", st.session_state.inputs['JPY'])
)
st.session_state.inputs['GBP'] = st.sidebar.number_input(
    "GBP", value=st.session_state.inputs['GBP'], on_change=update_input, args=("GBP", st.session_state.inputs['GBP'])
)

st.sidebar.header("Equity Market Indicators Regional Market Indices")
st.session_state.inputs['MXUS'] = st.sidebar.number_input(
    "MXUS", value=st.session_state.inputs['MXUS'], on_change=update_input, args=("MXUS", st.session_state.inputs['MXUS'])
)
st.session_state.inputs['MXEU'] = st.sidebar.number_input(
    "MXEU", value=st.session_state.inputs['MXEU'], on_change=update_input, args=("MXEU", st.session_state.inputs['MXEU'])
)
st.session_state.inputs['MXJP'] = st.sidebar.number_input(
    "MXJP", value=st.session_state.inputs['MXJP'], on_change=update_input, args=("MXJP", st.session_state.inputs['MXJP'])
)
st.session_state.inputs['MXBR'] = st.sidebar.number_input(
    "MXBR", value=st.session_state.inputs['MXBR'], on_change=update_input, args=("MXBR", st.session_state.inputs['MXBR'])
)
st.session_state.inputs['MXCN'] = st.sidebar.number_input(
    "MXCN", value=st.session_state.inputs['MXCN'], on_change=update_input, args=("MXCN", st.session_state.inputs['MXCN'])
)

st.sidebar.header("Economic Sentiment and Rates Interest Rates")
st.session_state.inputs['USGG30YR'] = st.sidebar.number_input(
    "USGG30YR", value=st.session_state.inputs['USGG30YR'], on_change=update_input, args=("USGG30YR", st.session_state.inputs['USGG30YR'])
)
st.session_state.inputs['USGG3M'] = st.sidebar.number_input(
    "USGG3M", value=st.session_state.inputs['USGG3M'], on_change=update_input, args=("USGG3M", st.session_state.inputs['USGG3M'])
)
st.session_state.inputs['EONIA'] = st.sidebar.number_input(
    "EONIA", value=st.session_state.inputs['EONIA'], on_change=update_input, args=("EONIA", st.session_state.inputs['EONIA'])
)

st.sidebar.header("Volatility Indicators")
st.session_state.inputs['VIX'] = st.sidebar.number_input(
    "VIX", value=st.session_state.inputs['VIX'], on_change=update_input, args=("VIX", st.session_state.inputs['VIX'])
)
st.session_state.inputs['BDIY'] = st.sidebar.number_input(
    "BDIY", value=st.session_state.inputs['BDIY'], on_change=update_input, args=("BDIY", st.session_state.inputs['BDIY'])
)

st.write("### User Input Summary")
st.dataframe(pd.DataFrame([st.session_state.inputs]))

if (st.button("Predict Market Crash", on_click=callback) or st.session_state.btn_clicked):
    input_df = pd.DataFrame([st.session_state.inputs])
    
    # Normalize the user input using the preloaded scaler
    normalized_input = scaler.transform(input_df[['VIX','BDIY', 'LUMSTRUU', 'LUACTRUU', 'LF94TRUU','XAU BGNL', 'CRY', 'DXY', 'JPY', 'GBP','MXUS', 'MXEU', 'MXJP', 'MXBR', 'MXCN','USGG30YR','USGG3M', 'EONIA']])
    
    st.session_state.final_result = make_prediction(pd.DataFrame(normalized_input, columns=input_df.columns))

    st.title("Market Crash Prediction Results")

    overall_prediction = "Crash" if st.session_state.final_result["overall_pred"] == 1 else "No Crash"
    st.header(f"Overall Prediction: {overall_prediction}")

    for strategy_data in st.session_state.final_result["Strategies"]:
        strategy_name = strategy_data["Strategy"]
        strategy_prediction = "Crash" if strategy_data["Strategy_prediction"] == 1 else "No Crash"

        with st.expander(f"Metric: {strategy_name} - Prediction: {strategy_prediction}"):
            st.subheader("Model Predictions")
            st.write(f"Indexes : {strategy_data['indexes']}")
            for model_name, model_prediction in strategy_data.items():
                if model_name not in ["Strategy", "Strategy_prediction"]:
                    model_result = "Crash" if model_prediction == 1 else "No Crash"
                    st.write(f"- {model_name}: {model_result}")

    st.subheader("AI Investment strategy")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": perform_rag("Give me best invest ment stregy based on given data.", st.session_state.final_result)}
        ]

    # Display existing messages with unique keys
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for user query
    if user_input := st.chat_input("Enter your question:", key="user_input"):
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process the user query with perform_rag
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")

            # Call perform_rag to get the assistant's response
            response = perform_rag(user_input, st.session_state.final_result)

            # Display the assistant's response
            response_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

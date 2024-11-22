# from flask import Flask, request, render_template
# import pickle
# import openai
# import numpy as np
# # Initialize Flask app
# application = Flask(__name__)
# app = application

# # Load your Parkinson's disease model
# model = pickle.load(open("Parkinson_disease.pkl", "rb"))

# # Set up your OpenAI API key
# openai.api_key = 'your_openai_api_key'

# # Define a function for chatbot interaction
# def chat_with_gpt(query):
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=f"The user asked: {query}. Provide a response related to Parkinson's disease.",
#         max_tokens=150
#     )
#     return response.choices[0].text.strip()

# # Route for prediction and chatbot
# @app.route('/', methods=['GET', 'POST'])
# def predict_datapoint():
#     result = ""
#     chatbot_response = ""

#     if request.method == 'POST':
#         # Check if the user wants a Parkinson's prediction or a chatbot response
#         if 'MDVP_Fo_Hz' in request.form:  # Prediction form
#             # Get the input values for the prediction model
#             MDVP_Fo_Hz = float(request.form.get("MDVP_Fo_Hz"))
#             MDVP_Fhi_Hz = float(request.form.get("MDVP_Fhi_Hz"))
#             MDVP_Flo_Hz = float(request.form.get("MDVP_Flo_Hz"))
#             MDVP_Jitter_percent = float(request.form.get("MDVP_Jitter_percent"))
#             MDVP_Jitter_Abs = float(request.form.get("MDVP_Jitter_Abs"))
#             MDVP_RAP = float(request.form.get("MDVP_RAP"))
#             MDVP_PPQ = float(request.form.get("MDVP_PPQ"))
#             Jitter_DDP = float(request.form.get("Jitter_DDP"))
#             MDVP_Shimmer = float(request.form.get("MDVP_Shimmer"))
#             MDVP_Shimmer_dB = float(request.form.get("MDVP_Shimmer_dB"))
#             Shimmer_APQ3 = float(request.form.get("Shimmer_APQ3"))
#             Shimmer_APQ5 = float(request.form.get("Shimmer_APQ5"))
#             MDVP_APQ = float(request.form.get("MDVP_APQ"))
#             Shimmer_DDA = float(request.form.get("Shimmer_DDA"))
#             NHR = float(request.form.get("NHR"))
#             HNR = float(request.form.get("HNR"))
#             RPDE = float(request.form.get("RPDE"))
#             DFA = float(request.form.get("DFA"))
#             spread1 = float(request.form.get("spread1"))
#             spread2 = float(request.form.get("spread2"))
#             D2 = float(request.form.get("D2"))
#             PPE = float(request.form.get("PPE"))
           
#             predict_data = [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
#                     MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
#                     Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2,
#                     D2, PPE]]
            
#             predict = model.predict(predict_data)
            
#             if predict[0] == 1:
#                 result = 'Person Has No Parkinson Disease'
#             else:
#                 result = 'Person Has Parkinson Disease'
            
#         elif 'chat_query' in request.form:  # Chatbot form
#             user_query = request.form.get("chat_query")
#             chatbot_response = chat_with_gpt(user_query)

#         return render_template('home.html', result=result, chatbot_response=chatbot_response)

#     else:
#         return render_template('home.html')


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", debug=True)

from flask import Flask, request, render_template
import pickle
import openai
import numpy as np

# Initialize Flask app
application = Flask(__name__)
app = application

# Load your Parkinson's disease model
model = pickle.load(open("Parkinson_disease.pkl", "rb"))

# Set up your OpenAI API key
openai.api_key = 'your_openai_api_key'

# Define a function for chatbot interaction
def chat_with_gpt(query):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"The user asked: {query}. Provide a response related to Parkinson's disease.",
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Route for prediction and chatbot
@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""
    chatbot_response = ""

    if request.method == 'POST':
        # Check if the user wants a Parkinson's prediction or a chatbot response
        if 'MDVP_Fo_Hz' in request.form:  # Prediction form
            # Get the input values for the prediction model
            MDVP_Fo_Hz = float(request.form.get("MDVP_Fo_Hz"))
            MDVP_Fhi_Hz = float(request.form.get("MDVP_Fhi_Hz"))
            MDVP_Flo_Hz = float(request.form.get("MDVP_Flo_Hz"))
            MDVP_Jitter_percent = float(request.form.get("MDVP_Jitter_percent"))
            MDVP_Jitter_Abs = float(request.form.get("MDVP_Jitter_Abs"))
            MDVP_RAP = float(request.form.get("MDVP_RAP"))
            MDVP_PPQ = float(request.form.get("MDVP_PPQ"))
            Jitter_DDP = float(request.form.get("Jitter_DDP"))
            MDVP_Shimmer = float(request.form.get("MDVP_Shimmer"))
            MDVP_Shimmer_dB = float(request.form.get("MDVP_Shimmer_dB"))
            Shimmer_APQ3 = float(request.form.get("Shimmer_APQ3"))
            Shimmer_APQ5 = float(request.form.get("Shimmer_APQ5"))
            MDVP_APQ = float(request.form.get("MDVP_APQ"))
            Shimmer_DDA = float(request.form.get("Shimmer_DDA"))
            NHR = float(request.form.get("NHR"))
            HNR = float(request.form.get("HNR"))
            RPDE = float(request.form.get("RPDE"))
            DFA = float(request.form.get("DFA"))
            spread1 = float(request.form.get("spread1"))
            spread2 = float(request.form.get("spread2"))
            D2 = float(request.form.get("D2"))
            PPE = float(request.form.get("PPE"))
           
            predict_data = [[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
                             MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3,
                             Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2,
                             D2, PPE]]
            
            predict = model.predict(predict_data)
            
            if predict[0] == 1:
                result = 'Person Has No Parkinson Disease'
            else:
                result = 'Person Has Parkinson Disease'
            
        elif 'chat_query' in request.form:  # Chatbot form
            user_query = request.form.get("chat_query")
            chatbot_response = chat_with_gpt(user_query)

        return render_template('home.html', result=result, chatbot_response=chatbot_response)

    else:
        return render_template('home.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/va')
def va():
    return render_template('va.html')


@app.route('/doctor')
def doctor():
    return render_template('doctor.html')


@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/statistics')  # Fixed route for statistics
def statistics():  # Fixed function name for statistics
    return render_template('statistics.html')  # Make sure this template exists

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

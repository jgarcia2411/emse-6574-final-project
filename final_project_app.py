############
# Built-In #
############
import re
import string
import time

###############
# Third-Party #
###############
import en_core_web_sm
from joblib import load
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import spacy

st.title('Feed Yourself!')
st.sidebar.markdown('''
    # EMSE 6574 - Final Project

    Put in a list of ingredients and we will predict the cuisine and 
    the number of calories based on the ingredients! Watch your caloric intake
    AND learn something about your food!

    ## Cuisine Type Model
    <INSERT DESCRIPTION HERE>

    ## Calories Model
    The way the calories model works is by taking the ingredients, preprocessing
    them using the Python `spaCy` library, and passing it into a `scikit-learn`
    pipeline consisting of a term-frequency vectorizer and random forest regressor 
    model.

    ---
''')

st.markdown('''
    Put a list of ingredients in the box below and we'll get to work...
''')
ingredients = st.text_area('Ingredients')

if st.button('Predict'):
    if ingredients == '':
        st.markdown('''
            Ingredients list is empty! Please put some ingredients in.
        ''')
    else:
        with st.spinner('Getting predictions...'):
            progress_bar = st.progress(0)

            # Get the appropriate spaCy model to use.
            spacy_model = en_core_web_sm.load(disable = ['parser', 'ner'])

            ################################
            # TEXT PREPROCESSING FUNCTIONS #
            ################################
            def _remove_punctuation(text, step):
                if step == 'initial':
                    return [
                        token for token in text if re.sub(r'[\W_]+', ' ', token.text)
                        not in string.punctuation
                        and re.sub(r'([\W_])+', ' ', token.text) != ' '
                        and re.sub(r'([\W_])+', ' ', token.text) != ''
                    ]
                elif step == 'last':
                    return [re.sub(r'[\W_]+', ' ', token) for token in text]

            def _remove_stop_words(text):
                return [token for token in text if not token.is_stop]

            def _lemmatize(text):
                return [token.lemma_ for token in text]

            def _remove_short_words(text):
                return [token for token in text if len(token) > 2]

            def preprocess_text(text, model):
                # Lowercase text and remove extra spaces.
                step_1_2 = ' '.join(
                    [word.lower() for word in str(text).split()]
                )
                
                if model == 'cuisine':
                    # Remove numbers
                    step_1_2 = re.sub('[0-9]', '', step_1_2)

                # Remove units.
                step_3 = re.sub(
                    '(oz|ounces|ounce|pound|pounds|lb|lbs|inch|inches|kg|cup|cups|tablespoon|teaspoon|tablespoons|teaspoons)', 
                    '', 
                    step_1_2
                )

                # Tokenize text with spaCy.
                step_4 = spacy_model(step_3)

                # Remove punctuation.
                step_5 = _remove_punctuation(step_4, step = 'initial')

                # Remove stop words.
                step_6 = _remove_stop_words(step_5)

                # Lemmatize text.
                step_7 = _lemmatize(step_6)

                # Remove punctuation again.
                step_8 = _remove_punctuation(step_7, step = 'last')

                # Remove words two characters or less.
                step_9 = _remove_short_words(step_8)

                # Remake sentence with new cleaned up tokens.
                return ' '.join(step_9)

            # Start processing.
            ingredients_processed_cuisine = preprocess_text(
                ingredients, 
                'cuisine'
            )
            progress_bar.progress(0.125)

            ingredients_processed_calories = preprocess_text(
                ingredients, 
                'calories'
            )
            progress_bar.progress(0.25)

            # Load cuisine type model.
            cuisine_model = load('cuisine_model.joblib')
            progress_bar.progress(0.50)

            # Load calories model.
            calories_model = load('calories_model.joblib')
            progress_bar.progress(0.75)

            # Get predictions.
            cuisine = cuisine_model.predict([ingredients_processed_cuisine])[0]
            calories = calories_model.predict([ingredients_processed_calories])[0]
            progress_bar.progress(1.0)

        st.balloons()

    # Reformat cuisine label
    cuisine_label = ' '.join([text.capitalize() for text in cuisine.split('_')])
    if 'Us' in cuisine_label:
        cuisine_label = 'Southern US'

    st.markdown(f'''
        Hm...we predict these ingredients to have **{round(calories, 2)}** calories and to be
        **{cuisine_label}** cuisine. Yum!
    ''')

    zoom = 5 # Setting a default zoom level.
    if cuisine == 'italian':
        latitude = 41.871941
        longitude = 12.567380
    elif cuisine == 'mexican':
        latitude = 23.634501
        longitude = -102.552788
    elif cuisine == 'southern_us':
        latitude = 37.090240
        longitude = -95.712891
    elif cuisine == 'indian':
        latitude = 20.593683
        longitude = 78.962883
    elif cuisine == 'chinese':
        latitude = 34.88593094075317
        longitude = 102.65625000000001
    elif cuisine == 'french':
        latitude = 46.98025235521883
        longitude = 1.9775390625000002
    elif cuisine == 'cajun_creole':
        latitude = 29.951066
        longitude = -90.071532
    elif cuisine == 'thai':
        latitude = 15.870032
        longitude = 100.992541
    elif cuisine == 'japanese':
        latitude = 36.204824
        longitude = 138.252924
    elif cuisine == 'greek':
        latitude = 39.074208
        longitude = 21.824312
    elif cuisine == 'spanish':
        latitude = 40.463667
        longitude = -3.74922
    elif cuisine == 'korean':
        latitude = 37.663998
        longitude = 127.978458
    elif cuisine == 'vietnamese':
        latitude = 14.058324
        longitude = 108.277199
    elif cuisine == 'moroccan':
        latitude = 31.791702
        longitude = -7.09262
    elif cuisine == 'british':
        latitude = 55.378051
        longitude = -3.435973
    elif cuisine == 'filipino':
        latitude = 12.879721
        longitude = 121.774017
    elif cuisine == 'irish':
        latitude = 53.142367
        longitude = -7.692054
    elif cuisine == 'jamaican':
        latitude = 18.109581
        longitude = -77.297508
        zoom = 7
    elif cuisine == 'russian':
        latitude = 61.52401
        longitude = 105.318756
        zoom = 3
    elif cuisine == 'brazilian':
        latitude = -14.235004
        longitude = -51.92528
        zoom = 4
    else:
        latitude = 0
        longitude = 0

    st.pydeck_chart(
        pdk.Deck(
            map_style = 'mapbox://styles/mapbox/light-v9',
            initial_view_state = pdk.ViewState(
                latitude = latitude,
                longitude = longitude,
                zoom = zoom,
                pitch = 50,
            )
        )   
    )
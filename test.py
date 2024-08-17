import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("trained_model.sav", 'rb'))

def wine_predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 1:
        return 'Good Quality Wine'
    else:
        return 'Bad Quality Wine'

def main():
    st.title('Wine')
    st.image('pic.jpg', caption='Wanna Have Some')

    fa = st.text_input('Fixed Acidity (4.6 - 15.9)')
    va = st.text_input('Volatile Acidity (0.12 - 1.58)')
    ca = st.text_input('Critic Acid (0 - 1)')
    rs = st.text_input('Residual Sugar (0.9 - 15.5)')
    cl = st.text_input('Chlorides (0.012 - 0.611)')
    fso4 = st.text_input('Free Sulfur Dioxide (1 - 72)')
    ts04 = st.text_input('Total Sulfur Dioxide (6 - 289)')
    d = st.text_input('Density (0.99007 - 1.00369)')
    ph = st.text_input('pH (2.74 - 4.01)')
    s = st.text_input('Sulphates (0.33 - 2)')
    ol = st.text_input('Alcohol (8.4 - 14.9)')


    diagnosis = ''
    if st.button('Wine Test'):
        diagnosis = wine_predict([fa, va, ca, rs, cl, fso4, ts04, d, ph, s, ol])
        st.success(diagnosis)

if __name__ == '__main__':
    main()

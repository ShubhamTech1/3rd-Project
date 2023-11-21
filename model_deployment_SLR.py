
import pandas as pd
import streamlit as st 
# import numpy as np

from sqlalchemy import create_engine
import pickle

poly_model = pickle.load(open('poly_model.pkl', 'rb'))


def predict_AT(data,user,pw,db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")
       
    df = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\REGRESSION\SIMPLE LINEAR\delivery_time.csv" )     
    df1 = pd.DataFrame(df)    
    prediction = pd.DataFrame(poly_model.predict(df1), columns = ['Pred_Delivery_Time']) 
    
    final = pd.concat([prediction, data], axis = 1)
    final.to_sql('logistic_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index= False)

    return final



def main():
    st.title("logistic prediction")
    st.sidebar.title("Ecommerce product prediction") 
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">AT prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html = True)
    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            data = pd.read_csv(uploadedFile)
        except:
                try:
                    data = pd.read_excel(uploadedFile)
                except:      
                    data = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_AT(data, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        st.table(result.style.background_gradient(cmap=cm))#.set_precision(2))

if __name__=='__main__':
    main()



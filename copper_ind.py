import numpy as np
import pandas as pd
import streamlit as st 
from streamlit_option_menu import option_menu 

import re
import warnings as wr
wr.filterwarnings('ignore')

#from pandas_profiling import ProfileReport
#from ydata_profiling import ProfileReport
import sweetviz as sv
import codecs


import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

from PIL import Image
import io

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer


# ***** STREAMLIT PAGE ICON ***** 

icon = Image.open("C:/Users/mahes/Downloads/icon.png")
# SETTING PAGE CONFIGURATION...........
st.set_page_config(page_title='INDUSTRIAL COPPER',page_icon=icon,layout="wide")

html_temp = """
        <div style="background-color:#fb607f;padding:10px;border-radius:10px">
        <h1 style="color:white;text-align:center;">INDUSTRIAL COPPER ML MODEL PREDICTION</h1>
        </div>"""

# components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
components.html(html_temp)
style = "<style>h2 {text-align: center;}</style>"
style1 = "<style>h3 {text-align: left;}</style>"


selected = option_menu(None,
                       options = ["Home","Data View and EDA","Selling Price Predicton","Status Prediction"],
                       icons = ["house-door-fill","bar-chart-line-fill","bi-binoculars-fill","bi-binoculars-fill"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"container": {"width": "100%"},
                               "icon": {"color": "white", "font-size": "24px"},
                               "nav-link": {"font-size": "24px", "text-align": "center", "margin": "-2px"},
                               "nav-link-selected": {"background-color": "#480607"}})

df = pd.read_csv("C:/Users/mahes/OneDrive/Desktop/copper industrial pro/Copper_Set.csv")
df_new = df.copy()
def home_func():
    with st.expander("DATA VIEW"):
        st.dataframe(df)

def st_display_sweetviz(report_html, width=1000, height = 500):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page, width=width, height=height, scrolling=True)
            
        
def data_preprocessing(df1): 
    # dealing with data in wrong format
    # for categorical variables, this step is ignored
    # df = df[df['status'].isin(['Won', 'Lost'])]
    df1['item_date'] = pd.to_datetime(df1['item_date'], format='%Y%m%d', errors='coerce').dt.date
    df1['quantity tons'] = pd.to_numeric(df1['quantity tons'], errors='coerce')
    df1['customer'] = pd.to_numeric(df1['customer'], errors='coerce')
    df1['country'] = pd.to_numeric(df1['country'], errors='coerce')
    df1['application'] = pd.to_numeric(df1['application'], errors='coerce')
    df1['thickness'] = pd.to_numeric(df1['thickness'], errors='coerce')
    df1['width'] = pd.to_numeric(df1['width'], errors='coerce')
    df1['material_ref'] = df1['material_ref'].str.lstrip('0')
    df1['product_ref'] = pd.to_numeric(df1['product_ref'], errors='coerce')
    df1['delivery date'] = pd.to_datetime(df1['delivery date'], format='%Y%m%d', errors='coerce').dt.date
    df1['selling_price'] = pd.to_numeric(df1['selling_price'], errors='coerce')      
    
    # material_ref has large set of null values, so replacing them with unknown
    df1['material_ref'].fillna('unknown', inplace=True)
    # deleting the remaining null values as they are less than 1% of data which can be neglected
    df2 = df1.dropna()
    return df2
        
def show_shape():
    st.write(df.shape)   

def show_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

def show_values(df3):
    missing_values_count = df3.isnull().sum()
    st.table(missing_values_count)
    
def selling_price_prediction():
    # Define the possible values for the dropdown menus
    status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered',
                      'Offerable']
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    # Define the widgets for user input
    with st.form("my_form"):
        col1, col2, col3 = st.columns([5, 2, 5])
        with col1:
            st.write(' ')
            status = st.selectbox("Status", status_options, key=1)
            item_type = st.selectbox("Item Type", item_type_options, key=2)
            country = st.selectbox("Country", sorted(country_options), key=3)
            application = st.selectbox("Application", sorted(application_options), key=4)
            product_ref = st.selectbox("Product Reference", product, key=5)
        with col3:
            st.write(
                f'<h5 style="color:rgb(0, 153, 153,0.4);">NOTE: Min & Max given for reference, you can enter any value</h5>',
                unsafe_allow_html=True)
            quantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            thickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            width = st.text_input("Enter width (Min:1, Max:2990)")
            customer = st.text_input("customer ID (Min:12458, Max:30408185)")
            submit_button = st.form_submit_button(label="PREDICT SELLING PRICE")
            st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)

        flag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for i in [quantity_tons, thickness, width, customer]:
            if re.match(pattern, i):
                pass
            else:
                flag = 1
                break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("please enter a valid number space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

    if submit_button and flag == 0:
        import pickle

        with open(r"model.pkl", 'rb') as file:
            loaded_model = pickle.load(file)
        with open(r'scaler.pkl', 'rb') as f:
            scaler_loaded = pickle.load(f)

        with open(r"t.pkl", 'rb') as f:
            t_loaded = pickle.load(f)

        with open(r"s.pkl", 'rb') as f:
            s_loaded = pickle.load(f)

        new_sample = np.array([[np.log(float(quantity_tons)), application, np.log(float(thickness)), float(width),
                                country, float(customer), int(product_ref), item_type, status]])
        new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, ]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :green[Predicted selling price:] ', np.exp(new_pred))

def status_prediction():
    
    item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
    country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
    application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67.,
                           79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
    product = ['611112', '611728', '628112', '628117', '628377', '640400', '640405', '640665',
               '611993', '929423819', '1282007633', '1332077137', '164141591', '164336407',
               '164337175', '1665572032', '1665572374', '1665584320', '1665584642', '1665584662',
               '1668701376', '1668701698', '1668701718', '1668701725', '1670798778', '1671863738',
               '1671876026', '1690738206', '1690738219', '1693867550', '1693867563', '1721130331', '1722207579']

    with st.form("my_form1"):
        col1, col2, col3 = st.columns([5, 1, 5])
        with col1:
            cquantity_tons = st.text_input("Enter Quantity Tons (Min:611728 & Max:1722207579)")
            cthickness = st.text_input("Enter thickness (Min:0.18 & Max:400)")
            cwidth = st.text_input("Enter width (Min:1, Max:2990)")
            ccustomer = st.text_input("customer ID (Min:12458, Max:30408185)")
            cselling = st.text_input("Selling Price (Min:1, Max:100001015)")

        with col3:
            st.write(' ')
            citem_type = st.selectbox("Item Type", item_type_options, key=21)
            ccountry = st.selectbox("Country", sorted(country_options), key=31)
            capplication = st.selectbox("Application", sorted(application_options), key=41)
            cproduct_ref = st.selectbox("Product Reference", product, key=51)
            csubmit_button = st.form_submit_button(label="PREDICT STATUS")
            st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                        background-color: #009999;
                        color: white;
                        width: 100%;
                    }
                    </style>
                """, unsafe_allow_html=True)

        cflag = 0
        pattern = "^(?:\d+|\d*\.\d+)$"
        for k in [cquantity_tons, cthickness, cwidth, ccustomer, cselling]:
            if re.match(pattern, k):
                pass
            else:
                cflag = 1
                break

        if csubmit_button and cflag == 1:
            if len(k) == 0:
                st.write("please enter a valid number space not allowed")
            else:
                st.write("You have entered an invalid value: ", k)

        if csubmit_button and cflag == 0:
            import pickle

            with open(r"clsmodel.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)

            with open(r'cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)

            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            # Predict the status for a new sample
            # 'quantity tons_log', 'selling_price_log','application', 'thickness_log', 'width','country','customer','product_ref']].values, X_ohe
            new_sample = np.array([[np.log(float(cquantity_tons)), np.log(float(cselling)), capplication,
                                    np.log(float(cthickness)), float(cwidth), ccountry, int(ccustomer), int(cproduct_ref),
                                    citem_type]])
            new_sample_ohe = ct_loaded.transform(new_sample[:, [8]]).toarray()
            new_sample = np.concatenate((new_sample[:, [0, 1, 2, 3, 4, 5, 6, 7]], new_sample_ohe), axis=1)
            new_sample = cscaler_loaded.transform(new_sample)
            new_pred = cloaded_model.predict(new_sample)
            if new_pred == 1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')

    st.write(f'<h6 style="color:rgb(0, 153, 153,0.35);">App Created by UMAMAHESWARI S</h6>', unsafe_allow_html=True)        
    


####........ STREAMLIT CODING ........####    

if selected == "Home":
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">HOME</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)
    col1,col2 = st.columns(2)
    with col1:
            st.image(Image.open("C:\\Users\\mahes\\Downloads\\imageml.png"),width=600)
            st.markdown("## :red[Done by] : UMAMAHESWARI S")
            st.markdown(style,unsafe_allow_html=True)
            st.markdown(":red[Githublink](https://github.com/mahes101)")
                     
    with col2:
            st.header(':red[INDUSTRIAL COPPER ANALYSIS]')  
            st.markdown(style, unsafe_allow_html=True)    
            st.write("The copper industry deals with less complex data related to sales and pricing.")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[SKILLS OR TECHNOLOGIES]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("Python scripting, Data Preprocessing, Visualization, EDA, Streamlit")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[DOMAIN]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("Manufacturing")
            st.markdown(style1, unsafe_allow_html=True)
            st.header(':red[ML PREDICTION]')
            st.markdown(style, unsafe_allow_html=True)
            st.write("ML Regression model which predicts continuous variable ‘Selling_Price and ML Classification model which predicts Status: WON or LOST.")
            st.markdown(style1, unsafe_allow_html=True)
if selected == "Data View and EDA":
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">Data View and EDA</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)
    choice = st.sidebar.selectbox("Choose an option",["Data View","Automated EDA"])
    if choice == "Data View":
        home_func()
        st.subheader("Number of rows and columns")
        show_shape()
        st.subheader("Information of dataset")
        s = show_info(df)
        st.text(s)
        st.subheader("Missing values count of each columns")
        show_values(df)
    elif choice == "Automated EDA":
        data = data_preprocessing(df)
        #profile = ProfileReport(data)   
        #st_profile_report(profile) 
        report = sv.analyze(df)
        report.show_html()
        st_display_sweetviz("SWEETVIZ_REPORT.html")    
    else:
        pass    
    
if selected == "Selling Price Predicton":  
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">SELLING PRICE PREDICTION</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)
    selling_price_prediction()  
if selected == "Status Prediction":  
    html_temp = """
        <div style="background-color:#915c83;padding:10px;border-radius:10px">
        <h3 style="color:white;text-align:center;">STATUS PREDICTION</h3>
        </div>"""
    # components.html("<p style='color:red;'> Streamlit Components is Awesome</p>")
    components.html(html_temp)  
    status_prediction()
    
        
    
    
    

         
    
        

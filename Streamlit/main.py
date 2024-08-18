import streamlit as st
import pandas as pd
import os
# os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objs as go
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column

# HTML for styling
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Mall Customer Segmentation App</h2>
    
    </div>
    
    """
st.markdown(html_temp, unsafe_allow_html=True)

# Page configuration
#st.set_page_config(
    #page_title="Customer Segmentation app",
   #page_icon=":running_shirt_with_sash:",
   # page_icon=":shopping_trolley:",
    #layout="wide",
    #initial_sidebar_state="expanded")

alt.themes.enable("dark")
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)
#st.balloons()
# Title and Introduction
#st.title("Customer Segmentation App")
import os
# Use an absolute path
image_path = os.path.join(os.getcwd(), 'images', 'cs.png')
st.image(image_path)

#image_path = "images/cs.png"
#if os.path.exists(image_path):
    #st.image(image_path)
#else:
    #st.error(f"Image not found: {image_path}")


st.sidebar.title("Welcome to my Project")
st.sidebar.write("""
    Upload a CSV or Excel file containing customer data. This application uses KMeans clustering to analyze customer personality data.
""")
# tabs
#menu = ["Business Understanding", "Data Understanding","Data preparation","Modeling & Evaluation","Predict"] # , "BigData: Spark"
#choice = st.sidebar.selectbox('Menu', menu)

# Main Menu
    
st.subheader("Business Objective")
st.write("""
        ###### Customer segmentation is a fundamental task in marketing and customer relationship management. With the advancements in data analytics and machine learning, it is now possible to group customers into distinct segments with a high degree of precision, allowing businesses to tailor their marketing strategies and offerings to each segment's unique needs and preferences.
    
        ###### Through this customer segmentation, businesses can achieve:
        - **Personalization**: Tailoring marketing strategies to meet the unique needs of each segment.
        - **Optimization**: Efficient allocation of marketing resources.
        - **Insight**: Gaining a deeper understanding of the customer base.
        - **Engagement**: Enhancing customer engagement and satisfaction.
    
        ###### => Problem/Requirement: Utilize machine learning and data analysis techniques in Python to perform customer segmentation.
        """)
       # st.image("Customer-Segmentation.png", caption="Customer Segmentation", use_column_width=True)

tabs = st.tabs(["Data", "Analysis", "model"])
with tabs[0]:
 # File Upload
  
 uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx'])  
    
 if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith('csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('xlsx'):
        data = pd.read_excel(uploaded_file, engine='openpyxl')
     # Handling missing values
    st.subheader('Handling Missing Values')
    st.write("Original data shape:", data.shape)
    st.write("Number of missing values before handling:", data.isnull().sum().sum())
    
    # Drop rows with any NaN values
    data.dropna(inplace=True)

    # Verify data integrity post handling missing values
    st.write("Number of missing values after handling:", data.isnull().sum().sum())

    # Display Data
    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.write(data)

    # Data Preprocessing
    st.subheader('Data Preprocessing')
    columns = data.columns.tolist()

    # Convert string columns to integers
    for col in columns:
        if data[col].dtype == 'object':  # Check if column dtype is object (string)
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col])

    selected_columns = st.multiselect('Select columns for clustering', columns)

    if selected_columns:
        st.write(f"Selected columns for clustering: {selected_columns}")
        if len(selected_columns) >= 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[selected_columns])

            # Clustering
            st.subheader('Clustering')
            num_clusters = st.slider('Select number of clusters', 2, 10, 3)
            kmeans = KMeans(n_clusters=num_clusters)
            data['Cluster'] = kmeans.fit_predict(scaled_data)

            # Visualize Clusters
            st.subheader('Cluster Visualization')
            if len(selected_columns) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], hue=data['Cluster'], palette='viridis', ax=ax)
                st.pyplot(fig)
            else:
                st.write("Please select at least two columns for clustering visualization.")
        else:
            st.write("Please select at least two columns for clustering visualization.")
    else:
        st.write("Please select columns for clustering.")
   
       


    
with tabs[1]:
    st.write("Number of Customers by Age Groups")
    df=pd.read_csv("Mall_Customers.csv")
    # Categorize ages into different groups
    age_18_25 = df.Age[(df.Age >= 18) & (df.Age <= 25)]
    age_26_35 = df.Age[(df.Age >= 26) & (df.Age <= 35)]
    age_36_45 = df.Age[(df.Age >= 36) & (df.Age <= 45)]
    age_46_55 = df.Age[(df.Age >= 46) & (df.Age <= 55)]
    age_55above = df.Age[df.Age >= 56]
    # Labels and counts for the bar plot
    agex = ["18-25", "26-35", "36-45", "46-55", "55+"]
    agey = [len(age_18_25.values), len(age_26_35.values), len(age_36_45.values), len(age_46_55.values), len(age_55above.values)]
   # Create a DataFrame with the age groups and their corresponding counts
       # Create a DataFrame with the age groups and their corresponding counts
    age_data = pd.DataFrame({
        'Age Group': agex,
        'Number of Customers': agey
    })

   # Now you can plot the bar chart using Streamlit
    st.bar_chart(age_data.set_index('Age Group'))

    import plotly.figure_factory as ff
    # Convert the data to long-form
    long_df = pd.melt(df, id_vars='Gender', value_vars=['Annual Income (k$)'], var_name='Variable', value_name='Value')

   
    
    # Plot using Plotly's distribution plot
    male_data = df[df['Gender'] == 'Male']['Annual Income (k$)']
    female_data = df[df['Gender'] == 'Female']['Annual Income (k$)']
    
    # Create a distribution plot
    fig = ff.create_distplot([male_data, female_data], ['Male', 'Female'], show_hist=False, show_rug=False)
    fig.update_layout(title_text='Distribution Plot of Annual Income by Gender')
    
    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


   # Create the Seaborn count plot
    plt.figure(figsize=(15, 5))
    sns.countplot(y='Gender', data=df)
    plt.title('Count of Customers by Gender')
    plt.xlabel('Count')
    plt.ylabel('Gender')

    # Display the plot in Streamlit
    st.pyplot(plt)

   # Footer
with tabs[2]:
   df = pd.read_csv("Mall_Customers.csv")
    # Map Gender to numeric
   gender_mapping = {'Male': 0, 'Female': 1}
   df['Gender'] = df['Gender'].map(gender_mapping)
    # Fit the KMeans model
   kmeans_model = KMeans(n_clusters=3) # n_init=10
   df['clusters'] = kmeans_model.fit_predict(df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']])
        
        # Create a 3D scatter plot using Plotly
   figure = px.scatter_3d(df,
                            color='clusters',
                            x="Age",
                            y="Annual Income (k$)",
                            z="Spending Score (1-100)",
                            category_orders={"clusters": ["0", "1", "2", "3", "4"]})
            
            # Streamlit app structure
   st.title("KMeans Clustering Visualization")
   st.write("""
            ### 3D Scatter Plot of Clusters
            This plot visualizes the clusters formed using the KMeans algorithm based on Age, Annual Income, and Spending Score.
            """)
            
            # Display the Plotly figure in the Streamlit app
   st.plotly_chart(figure)


    
st.markdown("---")
st.write("© 2024 [pooja vekal]. All rights reserved.")

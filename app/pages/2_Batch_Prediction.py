import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title("💳 Credit Card Fraud Detector")
st.header("📁 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  if len(df.columns) > 7:
    df = df.iloc[:, :7]
  
  st.subheader("🔍 Preview of Uploaded Data")
  st.dataframe(df.head())

  try:
    # Scale and predict
    scaled_data = scaler.transform(df)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)[:, 1]
    
    df['Prediction'] = predictions
    df['Fraud Probability'] = probabilities
    
    st.subheader("📊 Results")
    st.dataframe(df)
    
    # After prediction
    labels = ['Not Fraud', 'Fraud']
    counts = df['Prediction'].value_counts().sort_index()
    counts.index = ['Not Fraud' if i == 0 else 'Fraud' for i in counts.index]
    
    # Visualize a pie chart
    fig, ax = plt.subplots()
    ax.pie(counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
    st.pyplot(fig)
    
    # Bar chart
    st.bar_chart(counts, horizontal=True)

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
      st.metric("Total Transactions", len(df))
    with col2:
      st.metric("Predicted Frauds", int((df['Prediction'] == 1).sum()))
    with col3:
      st.metric("Avg. Fraud Probability", f"{df['Fraud Probability'].mean():.2f}")
    
    # Download results
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction Results", csv, "predictions.csv", "text/csv")

  except Exception as e:
    st.error(f"⚠️ Error processing file: {e}")

# Offer CSV template
expected_columns = ['distance_from_home', 
                    'distance_from_last_transaction', 
                    'ratio_to_median_purchase_price', 
                    'repeat_retailer', 
                    'used_chip', 
                    'used_pin_number', 
                    'online_order']

st.markdown("📥 Need a template?")

template_df = pd.DataFrame(columns=expected_columns)
csv_template = template_df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV Template", csv_template, "template.csv", "text/csv")
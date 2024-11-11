import streamlit as st
from transformers import pipeline

summarizer = pipeline("summarization", model="ayush0205/medicalSummarizer")

import streamlit as st

st.title("🩺📋  Medi Summarizer")



def generate_response(input_text):
    st.info(summarizer("summarize: "+ input_text)[0]['summary_text'])


with st.form("my_form"):
    text = st.text_area(
        "Enter The Patient's Medical History:",
        "Sample DISCHARGE SUMMARY Patient Information: Name: J.S. Age: 77 years Gender: Female Chief Complaint: Extreme fatigue and occasional dizziness on rising, along with severe constipation. History of Present Illness: J.S. presented to our clinic in June 2005, complaining of extreme fatigue severe enough that it caused her to go to sleep briefly while driving. Additionally, she was experiencing occasional dizziness on rising in the morning and severe constipation. She reported that an ophthalmologist had found increased pressure in one eye and she was using eye drops prescribed for this. Medical History: J.S. had received radioactive iodine as a child and was being treated with synthetic T3. She has a history of osteoporosis which was diagnosed elsewhere. Treatment: J.S. was treated with orally administered nutritional supplements that included a multivitamin, 3 g of ascorbic acid (bowel tolerance), 300 mg of magnesium/potassium/aspartate, 250 mg of calcium with 166 mg magnesium in a combination tablet taken at bedtime, 5 mg of phytonadione (vitamin K1) because of a history of osteoporosis diagnosed elsewhere, 200 mg of lipoic acid and 150 mg of TTFD. She was also given a series of intravenous infusions. Hospital Course: Two months after the treatment, J.S. reported that she had more energy. In spite of this, there was still deterioration in her laboratory tests. One month later, her laboratory results had improved but then deteriorated again. Discharge Condition: J.S.'s laboratory test results had improved significantly following her treatment. Discharge Plan: J.S. is being discharged with the recommendation that she continue with her current treatment plan to maintain her laboratory test results. She is to follow-up with our clinic for her next visit. Discharge Medications: No medication changes were made during this hospital course. Discharge Instructions: Instructions were given to J.S. to continue with her current treatment plan and schedule her follow-up appointment with our clinic. She was also advised to follow a healthy diet and lifestyle to maintain her well-being. Follow-Up: J.S. is to schedule her next visit with our clinic for further evaluation and monitoring of her condition. Signed by: [Name of physician]",
    )
    submitted = st.form_submit_button("Summarize")
    if submitted:
        generate_response(text)

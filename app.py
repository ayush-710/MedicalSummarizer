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
        "Sample Hospital Course: The patient was a 74-year-old male admitted for altered mental status and non-responsiveness. On admission, the patient had an elevated blood glucose and anion gap, as well as hypernatremia, acute kidney injury, and hemoconcentration. Imaging studies showed a right lower lobe mass or consolidation with multiple mediastinal and hilar masses compatible with lymphadenopathy, a suprasellar mass, and nodular densities in the adrenal glands and kidneys. Additional investigations showed small lymph nodes in the parotid glands, a persistent primary respiratory alkalosis with secondary metabolic alkalosis, and endocrine abnormalities consistent with gonadotropin deficiency and hypothyroidism. Treatment: The patient was started on half normal saline and insulin for the hyperglycemia. The acute kidney injury improved by day 3, though the hypernatremia did not correct until the patient was started on desmopressin on day 7. Given the significant metastatic burden, the plan was to obtain a lung biopsy and conduct a high dexamethasone suppression test for ectopic ACTH production. However, the patient declined further workup or treatment and opted for hospice. Course: The patient was sent home on hospice and was advised to increase his water intake. The family was informed of the abnormal imaging findings and provided with the options for palliative care. The patient's prognosis was explained to the family, and they agreed to the plan for comfort measures. A follow-up appointment was scheduled with the primary care provider. Discharge instructions were provided to the family, and they were advised to follow up with the primary care provider regularly.",
    )
    submitted = st.form_submit_button("Summarize")
    if submitted:
        generate_response(text)

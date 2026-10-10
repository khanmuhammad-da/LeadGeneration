
import json

import pandas as pd
import requests
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Lead Generator",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------
# API KEYS
# ---------------------------------------------------------

SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

SERPAPI_URL = "https://serpapi.com/search.json"

GEMINI_MODEL = "gemini-2.5-flash-lite"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


# ---------------------------------------------------------
# FUNCTION 1: FETCH LEADS FROM SERPAPI
# ---------------------------------------------------------

def fetch_leads(industry, location):
    """Fetch business leads from Google Maps using SerpApi."""

    params = {
        "engine": "google_maps",
        "type": "search",
        "q": f"{industry} in {location}",
        "api_key": SERPAPI_KEY,
        "hl": "en",
    }

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()
    results = data.get("local_results", [])

    leads = []

    for result in results[:10]:
        lead = {
            "business_name": result.get("title", "N/A"),
            "website": result.get("website", "N/A"),
            "location": result.get("address", location),
            "description": result.get(
                "description",
                f"{result.get('type', industry)} business",
            ),
            "business_type": result.get("type", "N/A"),
            "rating": result.get("rating", "N/A"),
            "reviews": result.get("reviews", "N/A"),
        }

        leads.append(lead)

    return leads


# ---------------------------------------------------------
# FUNCTION 2: CLASSIFY LEAD USING GEMINI
# ---------------------------------------------------------

def classify_lead(lead, industry):
    """Classify a business lead as HOT or COLD using Gemini AI."""

    prompt = f"""
You are a B2B sales lead qualification expert.

Your task is to classify a business lead as HOT or COLD
based on its potential relevance to the target industry.

TARGET INDUSTRY:
{industry}

BUSINESS DETAILS:
- Business Name: {lead.get("business_name", "N/A")}
- Website: {lead.get("website", "N/A")}
- Location: {lead.get("location", "N/A")}
- Description: {lead.get("description", "N/A")}
- Business Type: {lead.get("business_type", "N/A")}
- Rating: {lead.get("rating", "N/A")}
- Reviews: {lead.get("reviews", "N/A")}

CLASSIFICATION RULES:

HOT:
- The business appears highly relevant to the target industry.
- Available information indicates a plausible potential customer.
- The business's profile provides a reasonable basis for sales outreach.

COLD:
- The business appears unrelated to the target industry.
- Available information provides insufficient evidence of relevance.
- The business does not appear to be a plausible prospect.

IMPORTANT:
- Do not assume that a business is interested in buying.
- Do not invent purchasing intent, budget, or business needs.
- Base your decision only on the information provided.
- Return exactly one classification: HOT or COLD.
- Provide a short explanation for your decision.

Return valid JSON in this format:

{{
    "classification": "HOT",
    "reason": "The business appears relevant to the target industry."
}}
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "classification": {
                        "type": "STRING",
                        "enum": ["HOT", "COLD"],
                    },
                    "reason": {
                        "type": "STRING",
                    },
                },
                "required": [
                    "classification",
                    "reason",
                ],
            },
        },
    }

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
        },
        params={
            "key": GEMINI_API_KEY,
        },
        json=payload,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    response_text = (
        data["candidates"][0]["content"]["parts"][0]["text"]
    )

    result = json.loads(response_text)

    return result


# ---------------------------------------------------------
# FUNCTION 3: PROCESS ALL LEADS
# ---------------------------------------------------------

def process_leads(leads, industry):
    """Classify all leads and track processing progress."""

    final_results = []
    total = len(leads)

    progress_bar = st.progress(0)

    for index, lead in enumerate(leads):
        try:
            classification = classify_lead(
                lead,
                industry,
            )

            lead["classification"] = classification.get(
                "classification",
                "COLD",
            )

            lead["reason"] = classification.get(
                "reason",
                "No reason provided.",
            )

        except Exception as e:
            lead["classification"] = "COLD"
            lead["reason"] = (
                f"LLM classification failed: {str(e)}"
            )

        final_results.append(lead)

        progress_bar.progress(
            (index + 1) / total
        )

    progress_bar.empty()

    return final_results


# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.title("🎯 AI-Powered Lead Generation & Classification")

st.write(
    "Generate up to 10 business leads from Google Maps "
    "and use Gemini AI to classify each lead as HOT or COLD."
)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    industry = st.text_input(
        "🏢 Industry",
        placeholder="e.g. Solar Companies",
    )

with col2:
    location = st.text_input(
        "📍 Location",
        placeholder="e.g. Lahore, Pakistan",
    )


generate_button = st.button(
    "🚀 Generate Leads",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# GENERATE RESULTS
# ---------------------------------------------------------

if generate_button:

    if not industry.strip() or not location.strip():
        st.warning(
            "Please enter both industry and location."
        )

    else:

        try:

            # STEP 1: FETCH LEADS

            with st.spinner(
                "🔎 Searching for business leads..."
            ):
                leads = fetch_leads(
                    industry.strip(),
                    location.strip(),
                )

            if not leads:
                st.error(
                    "No business leads were found. "
                    "Try a different industry or location."
                )

            else:

                st.success(
                    f"Found {len(leads)} business leads."
                )

                # STEP 2: CLASSIFY LEADS

                with st.spinner(
                    "🤖 Gemini AI is classifying the leads..."
                ):
                    final_results = process_leads(
                        leads,
                        industry.strip(),
                    )

                # -----------------------------------------
                # RESULTS TABLE
                # -----------------------------------------

                st.subheader(
                    "📊 Lead Classification Results"
                )

                df = pd.DataFrame(final_results)

                df = df[
                    [
                        "business_name",
                        "website",
                        "location",
                        "description",
                        "business_type",
                        "rating",
                        "reviews",
                        "classification",
                        "reason",
                    ]
                ]

                df.columns = [
                    "Business Name",
                    "Website",
                    "Location",
                    "Description",
                    "Business Type",
                    "Rating",
                    "Reviews",
                    "Classification",
                    "AI Reason",
                ]

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

                # -----------------------------------------
                # SUMMARY
                # -----------------------------------------

                hot_count = sum(
                    1
                    for lead in final_results
                    if lead["classification"] == "HOT"
                )

                cold_count = sum(
                    1
                    for lead in final_results
                    if lead["classification"] == "COLD"
                )

                st.subheader("📈 Summary")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Leads",
                        len(final_results),
                    )

                with c2:
                    st.metric(
                        "🔥 HOT Leads",
                        hot_count,
                    )

                with c3:
                    st.metric(
                        "❄️ COLD Leads",
                        cold_count,
                    )

                # -----------------------------------------
                # DOWNLOAD RESULTS
                # -----------------------------------------

                csv = df.to_csv(
                    index=False,
                ).encode("utf-8-sig")

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="ai_leads.csv",
                    mime="text/csv",
                )

        except requests.exceptions.HTTPError as e:
            st.error(
                f"API request failed: {e}"
            )

        except requests.exceptions.RequestException as e:
            st.error(
                f"Network error while contacting an API: {e}"
            )

        except Exception as e:
            st.error(
                f"Something went wrong: {e}"
            )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "Lead data: SerpApi / Google Maps | "
    "Classification: Google Gemini"
)
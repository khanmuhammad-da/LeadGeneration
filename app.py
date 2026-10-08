import streamlit as st
import requests
import pandas as pd


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Lead Generator",
    page_icon="🎯",
    layout="wide"
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

    params = {
        "engine": "google_maps",
        "type": "search",
        "q": f"{industry} in {location}",
        "api_key": SERPAPI_KEY,
        "hl": "en"
    }

    response = requests.get(
        SERPAPI_URL,
        params=params,
        timeout=30
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
                f"{result.get('type', industry)} business"
            ),
            "business_type": result.get("type", "N/A"),
            "rating": result.get("rating", "N/A"),
            "reviews": result.get("reviews", "N/A")
        }

        leads.append(lead)

    return leads


# ---------------------------------------------------------
# FUNCTION 2: CLASSIFY LEAD USING GEMINI
# ---------------------------------------------------------

def classify_lead(lead, industry):

    prompt = f"""
You are an AI sales lead classification assistant.

We are looking for potential business leads in the following industry:

Industry: {industry}

Analyze the following business information.

Business Name:
{lead["business_name"]}

Business Type:
{lead["business_type"]}

Location:
{lead["location"]}

Website:
{lead["website"]}

Description:
{lead["description"]}

Rating:
{lead["rating"]}

Number of Reviews:
{lead["reviews"]}

Classify this business as either:

HOT
or
COLD

Use the following general logic:

HOT:
- Strong relevance to the requested industry
- Established business presence
- Clear business activity or commercial potential
- Good online presence or other positive business signals

COLD:
- Weak relevance to the industry
- Limited information
- Weak business presence
- Low apparent commercial relevance

Important:
This is a lead-potential classification based only on publicly available business information.
Do not claim that the business has expressed buying intent.

Return ONLY valid JSON in this exact format:

{{
    "classification": "HOT",
    "reason": "Short explanation in one sentence."
}}

The classification must be exactly HOT or COLD.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
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
                        "enum": ["HOT", "COLD"]
                    },
                    "reason": {
                        "type": "STRING"
                    }
                },
                "required": [
                    "classification",
                    "reason"
                ]
            }
        }
    }

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json"
        },
        params={
            "key": GEMINI_API_KEY
        },
        json=payload,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]

    import json

    result = json.loads(text)

    return result


# ---------------------------------------------------------
# FUNCTION 3: PROCESS ALL LEADS
# ---------------------------------------------------------

def process_leads(leads, industry):

    final_results = []

    progress_bar = st.progress(0)

    total = len(leads)

    for index, lead in enumerate(leads):

        try:

            classification = classify_lead(
                lead,
                industry
            )

            lead["classification"] = classification.get(
                "classification",
                "COLD"
            )

            lead["reason"] = classification.get(
                "reason",
                "No reason provided."
            )

        except Exception as e:

            lead["classification"] = "COLD"
            lead["reason"] = f"LLM classification failed: {str(e)}"

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
    "Generate 10 business leads from Google Maps and use Gemini AI "
    "to classify each lead as HOT or COLD."
)


# ---------------------------------------------------------
# INPUT SECTION
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    industry = st.text_input(
        "🏢 Industry",
        placeholder="e.g. Solar Companies"
    )

with col2:

    location = st.text_input(
        "📍 Location",
        placeholder="e.g. Lahore, Pakistan"
    )


generate_button = st.button(
    "🚀 Generate Leads",
    type="primary",
    use_container_width=True
)


# ---------------------------------------------------------
# GENERATE RESULTS
# ---------------------------------------------------------

if generate_button:

    if not industry or not location:

        st.warning(
            "Please enter both industry and location."
        )

    else:

        try:

            # Step 1: Fetch leads
            with st.spinner(
                "🔎 Searching for business leads..."
            ):

                leads = fetch_leads(
                    industry,
                    location
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

                # Step 2: Classify leads
                with st.spinner(
                    "🤖 Gemini AI is classifying the leads..."
                ):

                    final_results = process_leads(
                        leads,
                        industry
                    )

                # -------------------------------------------------
                # RESULTS
                # -------------------------------------------------

                st.subheader(
                    "📊 Lead Classification Results"
                )

                # Convert to dataframe
                df = pd.DataFrame(final_results)

                # Reorder columns
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
                        "reason"
                    ]
                ]

                # Rename columns
                df.columns = [
                    "Business Name",
                    "Website",
                    "Location",
                    "Description",
                    "Business Type",
                    "Rating",
                    "Reviews",
                    "Classification",
                    "AI Reason"
                ]

                # Display table
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # SUMMARY
                # -------------------------------------------------

                hot_count = sum(
                    1 for lead in final_results
                    if lead["classification"] == "HOT"
                )

                cold_count = sum(
                    1 for lead in final_results
                    if lead["classification"] == "COLD"
                )

                st.subheader("📈 Summary")

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Leads",
                        len(final_results)
                    )

                with c2:
                    st.metric(
                        "🔥 HOT Leads",
                        hot_count
                    )

                with c3:
                    st.metric(
                        "❄️ COLD Leads",
                        cold_count
                    )

                # -------------------------------------------------
                # DOWNLOAD
                # -------------------------------------------------

                csv = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="ai_leads.csv",
                    mime="text/csv"
                )

        except requests.exceptions.HTTPError as e:

            st.error(
                f"API request failed: {e}"
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
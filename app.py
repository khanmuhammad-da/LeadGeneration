
import json
import re

import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Lead Generator",
    page_icon="🎯",
    layout="wide"
)

SERPAPI_URL = "https://serpapi.com/search.json"

GEMINI_MODEL = "gemini-3.5-flash-lite"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    f"models/{GEMINI_MODEL}:generateContent"
)


# ============================================================
# LOAD API KEYS
# ============================================================

try:
    SERPAPI_KEY = st.secrets["SERPAPI_KEY"]
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

except Exception:
    st.error(
        "API keys are missing. Configure SERPAPI_KEY and "
        "GEMINI_API_KEY in your Streamlit Secrets."
    )
    st.stop()


# ============================================================
# FUNCTION 1: SEARCH FOR BUSINESS LEADS
# ============================================================

def search_leads(business_type, location, number_of_leads=10):
    """
    Retrieve business information from Google Maps through SerpApi.
    """

    params = {
        "engine": "google_maps",
        "q": f"{business_type} in {location}",
        "api_key": SERPAPI_KEY
    }

    try:
        response = requests.get(
            SERPAPI_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            st.error(f"SerpApi error: {data['error']}")
            return []

        places = data.get("local_results", [])
        leads = []

        for place in places[:number_of_leads]:

            leads.append({
                "Business Name": place.get("title", "N/A"),
                "Category": place.get("type", "N/A"),
                "Address": place.get("address", "N/A"),
                "Phone": place.get("phone", "N/A"),
                "Website": place.get("website", ""),
                "Rating": place.get("rating", "N/A"),
                "Reviews": place.get("reviews", "N/A"),
                "Lead Status": "UNCLASSIFIED",
                "AI Reason": "Not classified yet"
            })

        return leads

    except requests.exceptions.Timeout:
        st.error("Business search timed out. Please try again.")
        return []

    except requests.exceptions.HTTPError as error:
        st.error(f"SerpApi HTTP error: {error}")
        return []

    except requests.exceptions.RequestException as error:
        st.error(f"Business search failed: {error}")
        return []

    except (ValueError, TypeError):
        st.error("Could not process the business search results.")
        return []


# ============================================================
# FUNCTION 2: CLASSIFY A LEAD USING GEMINI
# ============================================================

def classify_lead(lead, target_customer, product_or_service):
    """
    Classify a business as HOT, WARM, or COLD based on the
    specified customer profile and available business information.

    API failures result in UNCLASSIFIED, not COLD.
    """

    prompt = f"""
You are a B2B sales lead qualification assistant.

OBJECTIVE
Evaluate whether this business is a potentially suitable
prospect for the product or service described below.

PRODUCT OR SERVICE:
{product_or_service}

TARGET CUSTOMER PROFILE:
{target_customer}

BUSINESS INFORMATION:
Business name: {lead.get("Business Name", "Unknown")}
Category: {lead.get("Category", "Unknown")}
Address: {lead.get("Address", "Unknown")}
Website: {lead.get("Website", "Unknown")}
Rating: {lead.get("Rating", "Unknown")}
Reviews: {lead.get("Reviews", "Unknown")}
Phone: {lead.get("Phone", "Unknown")}

CLASSIFICATION RULES

HOT:
The available information provides strong evidence that the
business matches the target customer profile and is a strong
prospect for the product or service.

WARM:
The business appears potentially relevant, but its suitability
or potential need requires further research.

COLD:
The available information suggests that the business is a poor
match for the target customer profile.

IMPORTANT:
1. Do not invent information about the business.
2. Do not claim the business intends to buy without evidence.
3. A high rating or many reviews alone does not make a lead HOT.
4. A missing website or phone number alone does not make a lead COLD.
5. Evaluate relevance to the specified product and customer profile.
6. If information is limited, use WARM when a plausible match exists.
7. Base your reason on the available information.
8. These classifications are preliminary estimates, not verified
   buying intent.

Return only valid JSON in this format:
{{
    "status": "HOT",
    "reason": "Brief explanation based on available information."
}}

The status must be exactly HOT, WARM, or COLD.
"""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

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
            "responseMimeType": "application/json"
        }
    }

    try:
        response = requests.post(
            GEMINI_URL,
            headers=headers,
            json=payload,
            timeout=45
        )

        response.raise_for_status()
        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            return {
                "status": "UNCLASSIFIED",
                "reason": "Gemini returned no classification."
            }

        parts = (
            candidates[0]
            .get("content", {})
            .get("parts", [])
        )

        if not parts:
            return {
                "status": "UNCLASSIFIED",
                "reason": "Gemini returned an empty response."
            }

        ai_text = parts[0].get("text", "").strip()

        # Remove Markdown code fences if returned by the model.
        ai_text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            ai_text,
            flags=re.IGNORECASE
        ).strip()

        result = json.loads(ai_text)

        status = str(
            result.get("status", "")
        ).strip().upper()

        reason = str(
            result.get("reason", "")
        ).strip()

        if status not in ["HOT", "WARM", "COLD"]:
            return {
                "status": "UNCLASSIFIED",
                "reason": "Gemini returned an invalid status."
            }

        if not reason:
            reason = "No explanation provided."

        return {
            "status": status,
            "reason": reason
        }

    except requests.exceptions.HTTPError:
        # Do not expose the API key or request URL.
        try:
            error_data = response.json()
            error_message = error_data.get(
                "error", {}
            ).get("message", "Gemini API request failed.")
        except (ValueError, AttributeError):
            error_message = (
                f"Gemini returned HTTP {response.status_code}."
            )

        return {
            "status": "UNCLASSIFIED",
            "reason": error_message
        }

    except requests.exceptions.Timeout:
        return {
            "status": "UNCLASSIFIED",
            "reason": "Gemini request timed out."
        }

    except requests.exceptions.RequestException:
        return {
            "status": "UNCLASSIFIED",
            "reason": "Network error while contacting Gemini."
        }

    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return {
            "status": "UNCLASSIFIED",
            "reason": "Could not interpret Gemini's response."
        }

    except Exception:
        return {
            "status": "UNCLASSIFIED",
            "reason": "Unexpected classification error."
        }


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🎯 AI Lead Generator")

st.markdown(
    """
    Discover businesses through Google Maps data and use AI to
    evaluate how closely each business matches your target customers.
    """
)

st.caption(
    "HOT/WARM/COLD are preliminary AI assessments, "
    "not confirmation of buying intent."
)


# ============================================================
# SIDEBAR: SEARCH AND CUSTOMER PROFILE
# ============================================================

st.sidebar.header("🔎 Lead Search")

business_type = st.sidebar.text_input(
    "Business Type",
    placeholder="e.g. Software companies"
)

location = st.sidebar.text_input(
    "Location",
    placeholder="e.g. Lahore, Pakistan"
)

number_of_leads = st.sidebar.slider(
    "Number of Leads",
    min_value=1,
    max_value=20,
    value=10
)

st.sidebar.divider()

st.sidebar.header("🎯 Target Customer Profile")

product_or_service = st.sidebar.text_area(
    "What are you selling?",
    placeholder=(
        "e.g. CRM software, digital marketing, "
        "web development services"
    )
)

target_customer = st.sidebar.text_area(
    "Which businesses are you targeting?",
    placeholder=(
        "e.g. Small and medium-sized businesses "
        "that need customer management software"
    )
)

classify_with_ai = st.sidebar.checkbox(
    "Classify leads using Gemini AI",
    value=True
)

search_button = st.sidebar.button(
    "🚀 Find Leads",
    use_container_width=True,
    type="primary"
)


# ============================================================
# SEARCH AND PROCESS LEADS
# ============================================================

if search_button:

    if not business_type.strip():
        st.warning("Please enter a business type.")

    elif not location.strip():
        st.warning("Please enter a location.")

    elif classify_with_ai and not product_or_service.strip():
        st.warning(
            "Please describe your product or service so AI "
            "can evaluate lead relevance."
        )

    elif classify_with_ai and not target_customer.strip():
        st.warning(
            "Please describe your target customer profile."
        )

    else:

        with st.spinner("Searching Google Maps for businesses..."):

            leads = search_leads(
                business_type=business_type.strip(),
                location=location.strip(),
                number_of_leads=number_of_leads
            )

        if not leads:
            st.warning(
                "No leads were returned. Try another search "
                "or check your SerpApi account."
            )

        else:

            st.success(f"Retrieved {len(leads)} business leads.")

            # ------------------------------------------------
            # AI CLASSIFICATION
            # ------------------------------------------------

            if classify_with_ai:

                progress = st.progress(0)
                progress_text = st.empty()

                total = len(leads)

                for index, lead in enumerate(leads):

                    progress_text.write(
                        f"Classifying {index + 1}/{total}: "
                        f"{lead['Business Name']}"
                    )

                    result = classify_lead(
                        lead=lead,
                        target_customer=target_customer,
                        product_or_service=product_or_service
                    )

                    lead["Lead Status"] = result["status"]
                    lead["AI Reason"] = result["reason"]

                    progress.progress(
                        (index + 1) / total
                    )

                progress_text.success(
                    "Classification process completed."
                )

            else:

                for lead in leads:
                    lead["Lead Status"] = "UNCLASSIFIED"
                    lead["AI Reason"] = (
                        "AI classification was disabled."
                    )

            # ------------------------------------------------
            # DATAFRAME
            # ------------------------------------------------

            df = pd.DataFrame(leads)

            # ------------------------------------------------
            # SUMMARY METRICS
            # ------------------------------------------------

            st.subheader("📊 Lead Summary")

            hot_count = int(
                (df["Lead Status"] == "HOT").sum()
            )

            warm_count = int(
                (df["Lead Status"] == "WARM").sum()
            )

            cold_count = int(
                (df["Lead Status"] == "COLD").sum()
            )

            unclassified_count = int(
                (df["Lead Status"] == "UNCLASSIFIED").sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🔥 HOT", hot_count)
            col2.metric("🌤️ WARM", warm_count)
            col3.metric("❄️ COLD", cold_count)
            col4.metric("⚠️ UNCLASSIFIED", unclassified_count)

            # ------------------------------------------------
            # FILTER RESULTS
            # ------------------------------------------------

            st.subheader("📋 Business Leads")

            status_filter = st.multiselect(
                "Filter by lead status",
                options=[
                    "HOT",
                    "WARM",
                    "COLD",
                    "UNCLASSIFIED"
                ],
                default=[
                    "HOT",
                    "WARM",
                    "COLD",
                    "UNCLASSIFIED"
                ]
            )

            filtered_df = df[
                df["Lead Status"].isin(status_filter)
            ]

            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Website": st.column_config.LinkColumn(
                        "Website",
                        display_text="Open Website"
                    )
                }
            )

            # ------------------------------------------------
            # DOWNLOAD CSV
            # ------------------------------------------------

            csv_data = filtered_df.to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                label="⬇️ Download Leads as CSV",
                data=csv_data,
                file_name="ai_generated_leads.csv",
                mime="text/csv"
            )

            # ------------------------------------------------
            # EXPLAIN CLASSIFICATION
            # ------------------------------------------------

            with st.expander(
                "ℹ️ How are HOT, WARM, and COLD determined?"
            ):

                st.markdown(
                    """
                    **HOT:** Strong evidence of a match with your
                    target customer profile.

                    **WARM:** Potential match, but more research is
                    needed.

                    **COLD:** Available information suggests a poor
                    match with your target customer profile.

                    **UNCLASSIFIED:** The API failed or the AI did
                    not return a valid classification.

                    **Important:** Google Maps information does not
                    establish a business's budget, purchasing plans,
                    or actual interest. Verify promising leads before
                    contacting them.
                    """
                )

else:

    st.info(
        "👈 Enter a business type, location, product/service, "
        "and target customer profile. Then click **Find Leads**."
    )

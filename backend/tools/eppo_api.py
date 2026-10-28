import os
import requests


# EPPO Global Database

def get_disease_info(disease_name: str) -> dict:
    """
    Query EPPO Global Database for disease information.
    Falls back to local knowledge base if API unavailable.
    """
    try:
        # EPPO API search
        search_url = "https://gd.eppo.int/taxon/search"
        params = {"kw": disease_name, "format": "json"}
        response = requests.get(search_url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data:
                eppo_code = data[0].get("eppocode", "")
                # Get detailed info
                detail_url = f"https://gd.eppo.int/taxon/{eppo_code}/datasheet"
                detail_response = requests.get(detail_url, timeout=10)
                return {
                    "source": "EPPO",
                    "disease": disease_name,
                    "eppo_code": eppo_code,
                    "url": f"https://gd.eppo.int/taxon/{eppo_code}",
                    "info": data[0]
                }
    except Exception:
        pass

    # Fallback to local knowledge base
    return _local_disease_info(disease_name)


def _local_disease_info(disease_name: str) -> dict:
    """Local knowledge base for common agricultural diseases."""
    diseases = {
        "wheat brown rust": {
            "pathogen": "Puccinia triticina",
            "symptoms": "Orange-brown pustules on leaves, circular to oval",
            "favorable_conditions": "Humidity > 80%, temperature 15-22°C",
            "spread": "Wind-dispersed spores",
            "economic_impact": "Yield loss up to 40% if untreated",
            "treatments": ["Tebuconazole", "Propiconazole", "Azoxystrobin"],
            "prevention": "Resistant varieties, crop rotation"
        },
        "tomato late blight": {
            "pathogen": "Phytophthora infestans",
            "symptoms": "Dark water-soaked lesions on leaves and stems",
            "favorable_conditions": "Humidity > 90%, temperature 10-25°C",
            "spread": "Rain splash, wind",
            "economic_impact": "Complete crop loss possible in 7-10 days",
            "treatments": ["Mancozeb", "Cymoxanil", "Metalaxyl"],
            "prevention": "Avoid overhead irrigation, good ventilation"
        },
        "olive fruit fly": {
            "pathogen": "Bactrocera oleae",
            "symptoms": "Punctures on olive fruit, larval tunneling",
            "favorable_conditions": "Temperature 20-30°C, humidity 50-70%",
            "spread": "Adult fly movement",
            "economic_impact": "Up to 80% fruit loss in severe infestations",
            "treatments": ["Dimethoate", "Deltamethrin", "Protein bait spray"],
            "prevention": "Monitoring traps, early harvest"
        },
        "wheat fusarium": {
            "pathogen": "Fusarium graminearum",
            "symptoms": "Bleached spikelets, pink-orange spore masses",
            "favorable_conditions": "Wet weather during flowering",
            "spread": "Rain splash during flowering",
            "economic_impact": "Yield loss 20-60%, mycotoxin contamination",
            "treatments": ["Tebuconazole", "Metconazole", "Prothioconazole"],
            "prevention": "Resistant varieties, crop rotation, early sowing"
        },
        "citrus greening": {
            "pathogen": "Candidatus Liberibacter asiaticus",
            "symptoms": "Yellowing leaves, lopsided fruit, bitter taste",
            "favorable_conditions": "Presence of psyllid vector",
            "spread": "Asian citrus psyllid (Diaphorina citri)",
            "economic_impact": "Tree death within 5 years, no cure",
            "treatments": ["Psyllid control: Imidacloprid, Spirotetramat"],
            "prevention": "Certified disease-free nursery stock, psyllid monitoring"
        }
    }

    key = disease_name.lower()
    for disease_key, info in diseases.items():
        if disease_key in key or key in disease_key:
            return {"source": "local", "disease": disease_name, **info}

    return {
        "source": "local",
        "disease": disease_name,
        "info": "Disease not found in local database. Consult agricultural extension service."
    }


# SMS Notification Service

def send_sms(phone_number: str, message: str) -> dict:
    """
    Send SMS notification to farmer.
    Uses Twilio API if configured, otherwise logs to console.
    """
    twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
    twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_from = os.getenv("TWILIO_PHONE_NUMBER")

    if twilio_sid and twilio_token and twilio_from:
        try:
            from twilio.rest import Client
            client = Client(twilio_sid, twilio_token)
            msg = client.messages.create(
                body=message,
                from_=twilio_from,
                to=phone_number
            )
            return {
                "success": True,
                "sid": msg.sid,
                "to": phone_number,
                "message": message
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    else:
        # Simulation mode: log to console
        print(f"[SMS SIMULATION] To: {phone_number}\nMessage: {message}")
        return {
            "success": True,
            "simulated": True,
            "to": phone_number,
            "message": message
        }

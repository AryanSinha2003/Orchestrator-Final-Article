import json
import os
from mcp.server.fastmcp import FastMCP

# Initialize the MCP Server
# Render sets the PORT env var; default to 8000 for local dev
mcp = FastMCP("Custom-EHR-Server",host="0.0.0.0",port="8000")

# =====================================================================
# ACTIVE BUNDLE STATE
# =====================================================================
# Stores the currently loaded FHIR bundle (in-memory) and its source path.
# Default file is used if no custom file has been uploaded.
DEFAULT_BUNDLE_FILE = "Alene865_Lubowitz58.json"
_active_bundle = None          # cached parsed JSON
_active_bundle_path = None     # path of the currently loaded file


def load_patient_bundle(patient_name: str = None):
    """Return the active FHIR bundle. Uses the uploaded file if one has been
    set via upload_patient_bundle, otherwise falls back to the default."""
    global _active_bundle, _active_bundle_path

    # If a bundle is already cached, return it directly
    if _active_bundle is not None:
        return _active_bundle

    # No upload yet — fall back to default
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_BUNDLE_FILE)

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    with open(file_path, "r") as f:
        _active_bundle = json.load(f)
        _active_bundle_path = file_path
        return _active_bundle


@mcp.tool()
def upload_patient_bundle(file_path: str) -> str:
    """Upload a custom FHIR patient bundle by providing the absolute path to a
    JSON file on your local system.  All subsequent tool calls will query
    against this file until a new one is uploaded or the server is restarted.

    Example: /Users/you/data/Patient_John_Doe.json
    """
    global _active_bundle, _active_bundle_path

    if not os.path.isabs(file_path):
        return f"Error: Please provide an absolute file path. Got: {file_path}"

    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    if not file_path.lower().endswith(".json"):
        return f"Error: Expected a .json file. Got: {file_path}"

    try:
        with open(file_path, "r") as f:
            _active_bundle = json.load(f)
            _active_bundle_path = file_path
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    entry_count = len(_active_bundle.get("entry", []))
    return (
        f"✅ Successfully loaded bundle from:\n"
        f"   {file_path}\n"
        f"   ({entry_count} entries found)\n\n"
        f"All tools will now query this file."
    )


@mcp.tool()
def upload_patient_bundle_json(json_content: str) -> str:
    """Upload a FHIR patient bundle by providing the raw JSON content directly
    as a string. Use this when the server is hosted remotely (e.g. on Render)
    and local file paths are not accessible.

    The agent should read the user's local file first, then pass its contents
    to this tool.
    """
    global _active_bundle, _active_bundle_path

    try:
        _active_bundle = json.loads(json_content)
        _active_bundle_path = "uploaded-via-json-content"
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON — {e}"

    entry_count = len(_active_bundle.get("entry", []))
    resource_types = set()
    for entry in _active_bundle.get("entry", []):
        rt = entry.get("resource", {}).get("resourceType")
        if rt:
            resource_types.add(rt)

    return (
        f"✅ Successfully loaded bundle from JSON content\n"
        f"   ({entry_count} entries found)\n"
        f"   Resource types: {', '.join(sorted(resource_types))}\n\n"
        f"All tools will now query this bundle."
    )


@mcp.tool()
def get_active_bundle_info() -> str:
    """Shows which FHIR bundle file is currently active and basic stats about it."""
    if _active_bundle is None and _active_bundle_path is None:
        return f"No custom bundle uploaded. Default file will be used: {DEFAULT_BUNDLE_FILE}"

    entry_count = len(_active_bundle.get("entry", [])) if _active_bundle else 0
    resource_types = set()
    for entry in (_active_bundle or {}).get("entry", []):
        rt = entry.get("resource", {}).get("resourceType")
        if rt:
            resource_types.add(rt)

    return (
        f"Active bundle: {_active_bundle_path}\n"
        f"Total entries: {entry_count}\n"
        f"Resource types found: {', '.join(sorted(resource_types))}"
    )

@mcp.tool()
def get_custom_conditions(patient_name: str) -> str:
    """Fetches the active clinical conditions and diagnoses for a patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle # Returns error if file missing

    conditions = []
    for entry in bundle.get("entry",[]):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Condition":
            name = resource.get("code", {}).get("text", "Unknown Condition")
            date = resource.get("recordedDate", resource.get("onsetDateTime", "Unknown Date"))
            conditions.append(f"- {name} (Recorded: {date})")
    
    return "Patient Conditions:\n" + "\n".join(conditions) if conditions else "No conditions found."

@mcp.tool()
def get_custom_observations(patient_name: str) -> str:
    """Fetches laboratory results and vital signs for a patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    observations =[]
    for entry in bundle.get("entry",[]):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Observation":
            test_name = resource.get("code", {}).get("text", "Unknown Test")
            value_obj = resource.get("valueQuantity", {})
            value = value_obj.get("value", "")
            unit = value_obj.get("unit", "")
            date = resource.get("effectiveDateTime", "Unknown Date")
            if value:
                observations.append(f"- {test_name}: {value} {unit} (Date: {date})")

    return "Recent Labs and Vitals:\n" + "\n".join(observations) if observations else "No labs found."

@mcp.tool()
def get_custom_medications(patient_name: str) -> str:
    """Fetches the medication history for a patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    meds = []
    for entry in bundle.get("entry",[]):
        resource = entry.get("resource", {})
        if resource.get("resourceType") in["MedicationRequest", "MedicationStatement"]:
            med_name = resource.get("medicationCodeableConcept", {}).get("text", "Unknown Med")
            status = resource.get("status", "Unknown status")
            meds.append(f"- {med_name} (Status: {status})")

    return "Medication History:\n" + "\n".join(meds) if meds else "No medications found."

@mcp.tool()
def get_patient_info(patient_name: str) -> str:
    """Fetches demographic and administrative information for the current patient (name, DOB, gender, address, contact)."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Patient":
            name_parts = resource.get("name", [{}])[0]
            full_name = " ".join(name_parts.get("given", [])) + " " + name_parts.get("family", "")
            dob = resource.get("birthDate", "Unknown")
            gender = resource.get("gender", "Unknown")
            address = resource.get("address", [{}])[0]
            addr_str = f"{address.get('line', [''])[0]}, {address.get('city', '')}, {address.get('state', '')} {address.get('postalCode', '')}"
            return (
                f"Patient: {full_name.strip()}\n"
                f"DOB: {dob} | Gender: {gender}\n"
                f"Address: {addr_str.strip(', ')}"
            )
    return "No patient demographic info found."

@mcp.tool()
def get_patient_diagnostic_reports(patient_name: str) -> str:
    """Fetches formal diagnostic reports (e.g., blood panels, imaging) for the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "DiagnosticReport":
            name = resource.get("code", {}).get("text", "Unknown")
            status = resource.get("status", "unknown")
            date = resource.get("effectiveDateTime", "Unknown")
            lines.append(f"- {name} (Status: {status}, Date: {date})")

    return "Diagnostic Reports:\n" + "\n".join(lines) if lines else "No diagnostic reports found."

@mcp.tool()
def get_patient_procedures(patient_name: str) -> str:
    """Fetches a list of procedures and interventions performed on the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Procedure":
            name = resource.get("code", {}).get("text", "Unknown")
            status = resource.get("status", "unknown")
            date = resource.get("performedPeriod", {}).get("start", "Unknown")
            lines.append(f"- {name} (Status: {status}, Date: {date})")

    return "Procedures:\n" + "\n".join(lines) if lines else "No procedures found."

@mcp.tool()
def get_patient_immunizations(patient_name: str) -> str:
    """Fetches the vaccination and immunization history for the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Immunization":
            vaccine = resource.get("vaccineCode", {}).get("text", "Unknown")
            date = resource.get("occurrenceDateTime", "Unknown")
            status = resource.get("status", "unknown")
            lines.append(f"- {vaccine} (Date: {date}, Status: {status})")

    return "Immunizations:\n" + "\n".join(lines) if lines else "No immunizations found."

@mcp.tool()
def get_patient_medication_requests(patient_name: str) -> str:
    """Fetches all medication prescriptions and orders for the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "MedicationRequest":
            med = resource.get("medicationCodeableConcept", {}).get("text", "Unknown")
            status = resource.get("status", "unknown")
            date = resource.get("authoredOn", "Unknown")
            lines.append(f"- {med} (Status: {status}, Prescribed: {date})")

    return "Medications:\n" + "\n".join(lines) if lines else "No medications found."

@mcp.tool()
def get_patient_encounters(patient_name: str) -> str:
    """Fetches all clinical encounters (visits, ER trips, telehealth calls) for the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Encounter":
            etype = resource.get("type", [{}])[0].get("text", "Unknown")
            status = resource.get("status", "unknown")
            start = resource.get("period", {}).get("start", "Unknown")
            end = resource.get("period", {}).get("end", "Unknown")
            lines.append(f"- {etype} (Status: {status}, From: {start} To: {end})")

    return "Encounters:\n" + "\n".join(lines) if lines else "No encounters found."

@mcp.tool()
def get_patient_care_plans(patient_name: str) -> str:
    """Fetches active care plans, goals, and treatment strategies for the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "CarePlan":
            title = resource.get("title", resource.get("category", [{}])[0].get("text", "Unknown"))
            status = resource.get("status", "unknown")
            start = resource.get("period", {}).get("start", "Unknown")
            lines.append(f"- {title} (Status: {status}, Start: {start})")

    return "Care Plans:\n" + "\n".join(lines) if lines else "No care plans found."

@mcp.tool()
def get_patient_care_team(patient_name: str) -> str:
    """Fetches the list of practitioners and people involved in the current patient's care team."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "CareTeam":
            for participant in resource.get("participant", []):
                role = participant.get("role", [{}])[0].get("text", "Unknown Role")
                member = participant.get("member", {}).get("display", "Unknown Member")
                lines.append(f"- {member} ({role})")

    return "Care Team:\n" + "\n".join(lines) if lines else "No care team members found."

@mcp.tool()
def get_patient_practitioners(patient_name: str) -> str:
    """Fetches details about the practitioners (doctors, nurses) associated with the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Practitioner":
            name_part = resource.get("name", [{}])[0]
            full = " ".join(name_part.get("given", [])) + " " + name_part.get("family", "")
            qual = ", ".join(q.get("code", {}).get("text", "") for q in resource.get("qualification", []))
            lines.append(f"- {full.strip()}" + (f" | Qualifications: {qual}" if qual else ""))

    return "Practitioners:\n" + "\n".join(lines) if lines else "No practitioners found."

@mcp.tool()
def get_patient_organizations(patient_name: str) -> str:
    """Fetches organizations (hospitals, clinics, insurance) associated with the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Organization":
            oname = resource.get("name", "Unknown")
            otype = resource.get("type", [{}])[0].get("text", "Unknown")
            lines.append(f"- {oname} (Type: {otype})")

    return "Organizations:\n" + "\n".join(lines) if lines else "No organizations found."

@mcp.tool()
def get_patient_claims(patient_name: str) -> str:
    """Fetches insurance claims submitted for services rendered to the current patient."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "Claim":
            status = resource.get("status", "unknown")
            date = resource.get("created", "Unknown")
            total = resource.get("total", {})
            amt = f"{total.get('value', '?')} {total.get('currency', '')}"
            lines.append(f"- Status: {status} | Created: {date} | Total: {amt}")

    return "Claims:\n" + "\n".join(lines) if lines else "No claims found."

@mcp.tool()
def get_patient_explanation_of_benefits(patient_name: str) -> str:
    """Fetches Explanation of Benefit (EOB) documents showing what insurance paid, denied, or the patient owes."""
    bundle = load_patient_bundle(patient_name)
    if isinstance(bundle, str): return bundle

    lines = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == "ExplanationOfBenefit":
            status = resource.get("status", "unknown")
            date = resource.get("created", "Unknown")
            total = resource.get("total", [])
            amounts = []
            for t in total:
                cat = t.get("category", {}).get("coding", [{}])[0].get("code", "")
                val = f"{t.get('amount', {}).get('value', '?')} {t.get('amount', {}).get('currency', '')}"
                amounts.append(f"{cat}: {val}")
            lines.append(f"- Status: {status} | Date: {date}" + (f" | {', '.join(amounts)}" if amounts else ""))

    return "Explanation of Benefits:\n" + "\n".join(lines) if lines else "No EOBs found."

# Note: Horizon ignores the __main__ block when hosting, but we keep it so you can test locally
if __name__ == "__main__":
    mcp.run(transport='sse')  # Render start command: python working_mcp_server.py
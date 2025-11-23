
def format_findings(findings: dict) -> str:
    """
    Formats the AI findings dictionary into a readable string list.
    """
    if not findings:
        return "Brak wykrytych patologii (No findings)."

    formatted_list = []
    for key, value in findings.items():
        if "base64" in key or "path" in key or "heatmap" in key:
            continue
        if isinstance(value, float):
            formatted_list.append(f"- {key}: {value:.1%} (prawdopodobieństwo)")
        else:
            formatted_list.append(f"- {key}: {value}")

    return "\n".join(formatted_list)

def format_patient_info(patient_info: dict) -> str:
    """
    Formats patient data into a string.
    """
    if not patient_info:
        return "Brak danych (N/A)"

    if isinstance(patient_info, dict):
        return f"""
        - Wiek: {patient_info.get('age', 'N/A')}
        - Płeć: {patient_info.get('gender', 'N/A')}
        - Choroby: {', '.join(patient_info.get('chronic_diseases', [])) or 'Brak'}
        - Alergie: {', '.join(patient_info.get('allergies', [])) or 'Brak'}
        """
    return str(patient_info)
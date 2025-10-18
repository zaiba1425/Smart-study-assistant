def categorize_text(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ["question", "quiz", "mcq"]):
        return "questions"
    elif any(word in text_lower for word in ["experiment", "lab", "practical"]):
        return "labs"
    elif any(word in text_lower for word in ["chapter", "topic", "lesson"]):
        return "textbook"
    else:
        return "notes"

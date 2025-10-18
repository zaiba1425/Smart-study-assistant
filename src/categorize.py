def categorize_text(text):
    """Categorizes text content based on keywords."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["question", "quiz", "mcq", "exam", "test"]):
        return "questions"
        
    elif any(word in text_lower for word in ["experiment", "lab", "practical", "procedure", "data analysis"]):
        return "labs"
        
    elif any(word in text_lower for word in ["chapter", "topic", "lesson", "definition", "introduction", "summary"]):
        return "textbook"
        
    else:
        return "notes"

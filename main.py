from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

client = Groq(api_key=GROQ_API_KEY)

# Request models
class CodeGenerationRequest(BaseModel):
    description: str

class CodeDebugRequest(BaseModel):
    code: str
    desired_outcome: str = None
    constraints: str = None

class DocumentationRequest(BaseModel):
    code: str

class ChatRequest(BaseModel):
    messages: list[dict]


def call_groq_api(prompt: str) -> str:
    """Call Groq Studio via the official Groq Python SDK."""
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Always respond in Markdown format."},
                {"role": "user", "content": prompt}
            ]
        )

        # Groq SDK returns choices with message content.
        if hasattr(chat_completion, "choices") and len(chat_completion.choices) > 0:
            choice = chat_completion.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return choice.message.content
            if hasattr(choice, "text"):
                return choice.text

        return "No response from API"
    except Exception as e:
        return f"Error: {str(e)}"

@app.post("/chat")
def chat(request: ChatRequest):
    """General conversational chat endpoint."""
    try:
        chat_completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=request.messages
        )

        if hasattr(chat_completion, "choices") and len(chat_completion.choices) > 0:
            choice = chat_completion.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                return {"message": choice.message.content}
            if hasattr(choice, "text"):
                return {"message": choice.text}

        return {"message": "No response from API"}
    except Exception as e:
        return {"error": str(e)}

@app.post("/code-generation")
def code_generation(request: CodeGenerationRequest):
    """
    Generate code based on description using role-based prompting and chain-of-thought.
    """
    prompt = f"""Act as a senior front-end developer. 

User Request: {request.description}

Please generate clean, well-structured code following these steps:
1. Analyze the requirements and break them down
2. Plan the structure and components needed
3. Generate the HTML, CSS, and JavaScript code
4. Ensure responsiveness and best practices
5. Add comments explaining key sections

Provide the complete code with proper formatting."""
    
    result = call_groq_api(prompt)
    return {"code": result, "description": request.description}

@app.post("/code-debugger")
def code_debugger(request: CodeDebugRequest):
    """
    Debug and refactor code with specific constraints.
    """
    outcome_text = f"\nDesired Outcome: {request.desired_outcome}" if request.desired_outcome else ""
    constraints_text = f"\nConstraints: {request.constraints}" if request.constraints else ""
    
    prompt = f"""Review the following code for bugs and suggest improvements.

Code to Review:
```
{request.code}
```{outcome_text}{constraints_text}

Please:
1. Identify any bugs or issues
2. Suggest efficiency improvements
3. Add clear comments explaining each function's purpose
4. Do not change the function names unless absolutely necessary
5. Provide the refactored code

Format your response with sections for "Bugs Found", "Improvements Made", and "Refactored Code"."""
    
    result = call_groq_api(prompt)
    return {"refactored_code": result, "original_code": request.code}

@app.post("/documentation-generator")
def documentation_generator(request: DocumentationRequest):
    """
    Generate comprehensive documentation using few-shot prompting.
    """
    prompt = f"""Generate comprehensive documentation for the following code snippet.

Code:
```
{request.code}
```

Use the following format as examples:

EXAMPLE 1:
/**
 * Calculates the sum of two numbers
 * @param {{number}} a - First number
 * @param {{number}} b - Second number
 * @returns {{number}} The sum of a and b
 */

EXAMPLE 2:
/**
 * Fetches user data from the API
 * @param {{string}} userId - The unique user identifier
 * @returns {{Promise<Object>}} User object with name, email, and profile
 * @throws {{Error}} If user not found
 */

Now generate similar comprehensive documentation for the provided code. Include:
1. Clear description of what the code does
2. @param tags for all parameters
3. @returns tag explaining return value
4. @throws tag if applicable
5. Usage examples if relevant"""
    
    result = call_groq_api(prompt)
    return {"documentation": result, "code_snippet": request.code}
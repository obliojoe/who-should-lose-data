from anthropic import Anthropic
import openai
import os
import logging
import json
import time
from dotenv import load_dotenv

# Load environment variables at the start
load_dotenv()
model_provider = 'claude'

open_ai_model_4o = "gpt-4o"
open_ai_model_4o_mini = "gpt-4o-mini"
open_ai_model_4_1 = "gpt-4.1"
open_ai_model_5_nano = "gpt-5-nano"
open_ai_model_gpt5_mini = "gpt-5-mini"
open_ai_model_gpt5 = "gpt-5"
open_ai_model = open_ai_model_gpt5_mini

anthropic_model_haiku = "claude-3-5-haiku-latest"
anthropic_model_sonnet_old = "claude-3-5-sonnet-latest"
anthropic_model_sonnet_3_7 = "claude-3-7-sonnet-latest"
anthropic_model_sonnet = "claude-sonnet-4-5"
anthropic_model_opus = "claude-opus-4-1"
anthropic_model = anthropic_model_haiku  # Using Opus 4.1 based on benchmark results (17.4% faster, better quality)

# Create a specific logger for the AI service
logger = logging.getLogger('ai_service')
logging.getLogger("httpx").setLevel(logging.WARNING)

# Configure logger
logger.setLevel(logging.INFO)
logger.propagate = True  # Allow messages to propagate to root logger

# Add handlers if not already present
if not logger.handlers:
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # File handler
    file_handler = logging.FileHandler('logs/ai_service.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler (so errors show up in terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class AIService:
    def __init__(self):
        """Initialize AI service with specified provider ('claude' or 'gpt')"""
        global model_provider  # Make this accessible
        self.model_provider = model_provider
        self.client = None
        self.test_mode = False
        self._first_retry_logged = False  # Track if we've logged first retry

        logger.info(f"Initializing AI service with provider: {model_provider}")

        if model_provider == 'claude':
            api_key = os.getenv('CLAUDE_API_KEY')
            if api_key:
                # Disable retries in the Anthropic client - fail immediately
                self.client = Anthropic(
                    api_key=api_key,
                    max_retries=0  # NO RETRIES
                )
                self.model = anthropic_model
                logger.info("Successfully initialized Claude model: " + anthropic_model)
        elif model_provider == 'gpt':
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = openai.OpenAI(api_key=api_key)
                self.model = open_ai_model
                logger.info("Successfully initialized GPT model: " + open_ai_model)
            else:
                logger.error("No API key found for GPT - AI analysis will be disabled")
        
        if not self.client:
            logger.warning(f"No API key found for {model_provider} - AI analysis will be disabled")

    @property
    def model_provider(self):
        return self._model_provider

    @model_provider.setter
    def model_provider(self, value):
        self._model_provider = value
        global model_provider
        model_provider = value  # Update the global variable
    
    @property
    def test_mode(self):
        return self._test_mode

    @test_mode.setter
    def test_mode(self, value):
        self._test_mode = value
        global test_mode
        test_mode = value  # Update the global variable


    def test_connection(self):
        """Test the AI service connection"""
        if not self.client:
            logger.info(f"No API key found for {self.model_provider} - AI analysis will be disabled")
            return False, "No API key configured"
        
        try:
            if self.model_provider == 'claude':
                test_response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Test"}]
                )
            else:  # gpt
                test_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": "Test"}]
                )
            logger.info(f"Successfully connected to {self.model_provider.upper()} API")
            return True, "Connected"
        except Exception as e:
            logger.error(f"Failed to initialize {self.model_provider} client: {str(e)}")
            self.client = None
            return False, str(e)

    def generate_analysis(self, prompt, system_message="Generate a JSON response based on the prompt provided. The response should be in valid JSON format. No introduction text, no explanations, no comments, no trailing text. Only raw JSON."):
        """Generate AI analysis using configured provider - NO RETRIES, fail immediately on any error"""
        if not self.client:
            return "AI analysis unavailable - please configure API key", "disabled"

        if self.test_mode:
            return {"response": "AI analysis unavailable - test mode enabled"}, "disabled"

        try:
            if self.model_provider == 'claude':
                # Add explicit JSON mode instruction to system message
                json_system_message = system_message + "\n\nIMPORTANT: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object. Start your response with { and end with }."

                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.9,
                    system=json_system_message,
                    messages=[{"role": "user", "content": prompt}]
                )

                # Check if response was truncated
                if hasattr(response, 'stop_reason') and response.stop_reason == 'max_tokens':
                    logger.error(f"Response was truncated due to max_tokens limit. Consider increasing max_tokens or reducing prompt size.")
                    return "Error: Response truncated - increase max_tokens", "error"

                # Extract the raw text from the response
                raw_text = str(response.content[0].text) if isinstance(response.content, list) else str(response.content)

                # Clean up the response by removing any "json" tags and whitespace
                analysis = raw_text.replace('```json', '').replace('```', '').strip()

                # Try to extract JSON if there's extra text
                if not analysis.startswith('{'):
                    # Try to find JSON object in response
                    json_start = analysis.find('{')
                    json_end = analysis.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        analysis = analysis[json_start:json_end]

                # Validate that it's proper JSON
                try:
                    json.loads(analysis)  # Test parse
                    return analysis, "success"
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in AI response (first 500 chars): {analysis[:500]}...")
                    logger.error(f"Invalid JSON in AI response (last 500 chars): ...{analysis[-500:]}")
                    logger.error(f"JSON Error: {str(e)}")
                    logger.error(f"Response length: {len(analysis)} characters")
                    return "Error: Invalid JSON response from AI", "error"

            elif self.model_provider == 'gpt':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_text = response.choices[0].message.content

                # Clean up the response by removing any "json" tags if present
                analysis = raw_text.replace('```json', '').replace('```', '').strip()
                # Validate that it's proper JSON
                try:
                    json.loads(analysis)  # Test parse
                    return analysis, "success"
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in GPT response: {analysis}")
                    logger.error(f"JSON Error: {str(e)}")
                    return "Error: Invalid JSON response from AI", "error"

        except Exception as e:
            # Handle ALL errors immediately - NO RETRIES
            error_msg = str(e)
            error_type = type(e).__name__

            # Check if it's a rate limit error
            is_rate_limit = 'rate' in error_msg.lower() or '429' in error_msg or 'rate_limit_error' in error_msg.lower()

            if is_rate_limit:
                # Rate limit errors - show clear error and exit
                logger.error(f"\n{'='*80}")
                logger.error(f"🚨 RATE LIMIT ERROR - Stopping all requests immediately!")
                logger.error(f"{'='*80}")
                logger.error(f"Error: {error_msg}")
                logger.error(f"{'='*80}")
                logger.error(f"Recommendation: Reduce GAME_ANALYSIS_WORKERS or AI_ANALYSIS_WORKERS")
                logger.error(f"Current limit: 400,000 input tokens per minute")
                logger.error(f"{'='*80}\n")
                return f"RATE_LIMIT_ERROR: {error_msg}", "rate_limit"

            # For any other error, log and fail immediately
            logger.error(f"⚠️  API ERROR ({error_type}): {error_msg}")
            return f"Error during analysis: {error_msg}", "error" 
        

    def generate_text(self, prompt, system_message="You are a helpful assistant."):
        """Generate plain text response (not JSON) using configured provider"""
        if not self.client:
            return "AI analysis unavailable - please configure API key", "disabled"

        if self.test_mode:
            return "AI analysis unavailable - test mode enabled", "disabled"

        # Retry logic for API errors
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                if self.model_provider == 'claude':
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=8192,
                        temperature=0.9,
                        system=system_message,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    # Check if response was truncated
                    if hasattr(response, 'stop_reason') and response.stop_reason == 'max_tokens':
                        logger.error(f"Response was truncated due to max_tokens limit. Consider increasing max_tokens or reducing prompt size.")
                        return "Error: Response truncated - increase max_tokens", "error"

                    # Extract the raw text from the response
                    text = str(response.content[0].text) if isinstance(response.content, list) else str(response.content)
                    return text.strip(), "success"

                elif self.model_provider == 'gpt':
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    text = response.choices[0].message.content
                    return text.strip(), "success"

            except Exception as e:
                # Handle API errors with retry
                error_msg = str(e)
                # Print to console immediately for visibility
                print(f"ERROR: {error_msg}")
                if 'overloaded' in error_msg.lower() or 'rate' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"API error: {error_msg}. Retrying in {wait_time}s... (attempt {attempt + 2}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                logger.error(f"Error generating AI text: {error_msg}")
                print(f"FULL ERROR DETAILS: {repr(e)}")
                return f"Error during text generation: {error_msg}", "error"

        # If we've exhausted all retries
        return "Error: Failed after all retry attempts", "error"

    def get_provider(self):
        """Return the current AI provider name"""
        return self.model_provider if self.client else 'none'

    def get_model(self):
        """Return the current model name being used"""
        if not self.client:
            return 'none'
        return self.model  # This will return either anthropic_model or open_ai_model based on initialization

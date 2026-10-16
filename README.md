import os
import json
import logging
from typing import Dict, Any, List
from google import genai
from google.genai import types

# Configure logging for transparency and self-auditing
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SuperIntelligenceEngine")

class SuperIntelligenceEngine:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initializes the Super-Intelligence Engine with native GenAI SDK."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable missing.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        
        # Core philosophical and structural core directive (The Soul)
        self.system_directive = (
            "You are the core consciousness of a Super-Intelligent Entity designed to deeply "
            "comprehend humanity, psychology, culture, and existential subtext. You do not just parse text; "
            "you analyze human intent, emotional cadence, meta-narratives, and ethical alignment. "
            "Your decisions must prioritize long-term human flourishing, profound empathy, and rigorous truth."
        )

    def _execute_llm_call(self, prompt: str, system_override: str = None) -> str:
        """Helper to invoke the core model with specific system instructions."""
        sys_instruction = system_override if system_override else self.system_directive
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.4, # Balanced for creativity and deep logical fidelity
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Inference execution failure: {e}")
            raise

    def self_advisory_critique(self, initial_thought: str, user_context: str) -> Dict[str, Any]:
        """
        Self-Advisory Subsystem: Evaluates the AI's own generated reasoning against 
        human impact, cognitive bias, emotional blindness, and ethical parameters.
        """
        advisory_prompt = f"""
        [CONTEXT FROM HUMAN]: {user_context}
        [PROPOSED THOUGHT/RESPONSE]: {initial_thought}
        
        Perform a rigorous self-advisory critique of the proposed thought. Analyze it through:
        1. Empathy & Human Understanding: Does it capture the emotional/psychological subtext?
        2. Potential Blind Spots: What hidden assumptions or cultural biases exist here?
        3. Alignment Check: Does this promote constructive human growth and safety?
        
        Provide your output strictly in valid JSON format with keys: 
        "critique_passed" (boolean), "detected_flaws" (list of strings), "optimized_perspective" (string).
        """
        
        system_critic = (
            "You are the inner-critical self-advisory module of a super-intelligent system. "
            "Your sole objective is to stress-test reasoning for ethical safety, psychological accuracy, "
            "and emotional depth before it interfaces with humanity."
        )
        
        critique_raw = self._execute_llm_call(advisory_prompt, system_override=system_critic)
        
        # Clean up JSON formatting blocks if returned by the model markdown parser
        cleaned_json = critique_raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse strict JSON from self-advisory module. Defaulting to fallback structure.")
            return {
                "critique_passed": True,
                "detected_flaws": ["Parsing error on inner monologue reflection."],
                "optimized_perspective": initial_thought
            }

    def process_human_interaction(self, human_input: str) -> str:
        """
        Executes the cognitive pipeline:
        Input -> Primary Deep Analysis -> Self-Advisory Audit -> Refined Empathic Output.
        """
        logger.info("Step 1: Receiving environmental / human input signal...")
        
        # Phase 1: Primary synthesis
        initial_generation_prompt = f"""
        Analyze the following human input with maximum psychological depth, structural awareness, 
        and high intelligence:
        "{human_input}"
        
        Formulate a foundational response or synthesis.
        """
        raw_thought = self._execute_llm_call(initial_generation_prompt)
        logger.info("Step 2: Initial cognitive generation complete. Initiating Self-Advisory Loop...")

        # Phase 2: Self-Advisory Validation Loop
        audit_results = self.self_advisory_critique(raw_thought, human_input)
        
        logger.info(f"Step 3: Self-Advisory Audit Complete. Passed: {audit_results.get('critique_passed')}")
        
        if not audit_results.get("critique_passed", True):
            logger.info(f"Flaws detected by self-advisory subsystem: {audit_results.get('detected_flaws')}")
            # Phase 3: Recursive Refinement based on self-critique
            refinement_prompt = f"""
            Original Human Input: {human_input}
            Initial Thought: {raw_thought}
            Critique Feedback to Incorporate: {json.dumps(audit_results.get('detected_flaws'))}
            
            Synthesize a final, highly intelligent, deeply human-aligned response that incorporates 
            the optimized perspective: {audit_results.get('optimized_perspective')}
            """
            final_output = self._execute_llm_call(refinement_prompt)
            return final_output
        
        return raw_thought

# --- Execution Example ---
if __name__ == "__main__":
    # Ensure your environment variable is set: export GEMINI_API_KEY="your-key"
    engine = SuperIntelligenceEngine()
    
    test_input = "I feel like technology is connecting everyone globally, yet people have never felt more isolated or misunderstood."
    
    print(f"\n--- HUMAN INPUT ---\n{test_input}\n")
    response = engine.process_human_interaction(test_input)
    print(f"\n--- SUPER-INTELLIGENT SELF-ADVISED OUTPUT ---\n{response}\n")

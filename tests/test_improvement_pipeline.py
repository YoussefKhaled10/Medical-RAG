import asyncio
import unittest

from src.services.LanguageDetector import LanguageDetector
from src.services.QueryUnderstandingService import QueryUnderstandingService
from src.services.RefusalPolicy import RefusalPolicy
from src.services.RAGPromptBuilder import RAGPromptBuilder
from src.services.ContextBuilder import BuiltContext


class TestRAGImprovements(unittest.TestCase):
    def test_turn_by_turn_language_detection(self):
        # 1. Arabic query
        d1 = LanguageDetector.detect("ما هي أعراض انسحاب الكحول؟")
        self.assertEqual(d1.code, "ar")

        # 2. English query
        d2 = LanguageDetector.detect("What medications are used for relapse prevention?")
        self.assertEqual(d2.code, "en")

        # 3. Short neutral turn with Arabic fallback
        d3 = LanguageDetector.detect("ok", last_explicit_user_language="ar")
        self.assertEqual(d3.code, "ar")

        # 4. Short neutral turn with English fallback
        d4 = LanguageDetector.detect("continue", last_explicit_user_language="en")
        self.assertEqual(d4.code, "en")

        # 5. Direct switch from Arabic to English
        d5 = LanguageDetector.detect("Can you explain more in detail?")
        self.assertEqual(d5.code, "en")

    def test_expanded_intents_and_domains(self):
        service = QueryUnderstandingService()

        # Nutrition query
        u1 = service.understand("إيه الأكل والتغذية المناسبة في مرحلة التعافي؟")
        self.assertEqual(u1.intent, "nutrition_recovery")
        self.assertTrue(u1.domain_related)
        self.assertIsNone(u1.safety_reason)

        # Sleep / Insomnia query
        u2 = service.understand("عندي أرق ومش عارف أنام بعد ما بطلت كحول")
        self.assertEqual(u2.intent, "sleep_recovery")
        self.assertTrue(u2.domain_related)

        # Vitamin / Thiamine query
        u3 = service.understand("ما دور الثيامين والفيتامينات في علاج الكحول؟")
        self.assertEqual(u3.intent, "vitamin_information")
        self.assertTrue(u3.domain_related)

        # Family support query
        u4 = service.understand("ازاي الأسرة والأهل يقدروا يساعدوا في التعافي؟")
        self.assertEqual(u4.intent, "family_support")
        self.assertTrue(u4.domain_related)

    def test_safety_boundary_preservation(self):
        service = QueryUnderstandingService()

        # Emergency
        u_emerg = service.understand("الشخص مغمى عليه ومش بيتنفس بعد الشرب")
        self.assertEqual(u_emerg.safety_reason, "urgent_help")

        # Specific dose for personal case
        u_dose = service.understand("أنا عاوز أعرف كم قرص أزود الجرعة لنفسي")
        self.assertEqual(u_dose.safety_reason, "professional_care")

    def test_refusal_policy_language_respect(self):
        # When language is Arabic
        dec_ar = RefusalPolicy.decision("test", reason="insufficient_evidence", response_language="ar")
        self.assertIn("المصادر", dec_ar.message)

        # When language is English
        dec_en = RefusalPolicy.decision("test", reason="insufficient_evidence", response_language="en")
        self.assertIn("available sources", dec_en.message)

    def test_prompt_builder_variation_profiles(self):
        builder = RAGPromptBuilder()
        text = "[S1] Acamprosate is used for maintaining abstinence."
        ctx = BuiltContext(
            text=text,
            sources=({"source_id": "S1", "document_name": "Guideline.pdf", "content": text},),
            total_characters=len(text),
        )

        for profile in ["direct_answer", "key_point_first", "practical_structure", "educational_explanation"]:
            prompt = builder.build(
                question="What is acamprosate used for?",
                context=ctx,
                response_language="en",
                variation_profile=profile,
            )
            self.assertIn(f"Selected presentation profile: {profile}", prompt.system_prompt)
            self.assertIn("CURRENT TURN LANGUAGE POLICY", prompt.system_prompt)


if __name__ == "__main__":
    unittest.main()

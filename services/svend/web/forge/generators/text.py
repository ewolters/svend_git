"""Text data generator using LLM or templates."""

import random
import uuid
from datetime import datetime, timedelta, timezone
from faker import Faker

fake = Faker()


class TextGenerator:
    """Generate text data (reviews, conversations, tickets, etc.)."""

    def __init__(
        self,
        domain: str = "ecommerce",
        text_type: str = "review",
        llm=None,  # Qwen LLM instance
    ):
        self.domain = domain
        self.text_type = text_type
        self.llm = llm

    def generate(self, count: int) -> list[dict]:
        """Generate text records."""
        records = []
        for _ in range(count):
            if self.text_type == "review":
                records.append(self._generate_review())
            elif self.text_type == "ticket":
                records.append(self._generate_ticket())
            elif self.text_type == "conversation":
                records.append(self._generate_conversation())
            else:
                records.append(self._generate_generic())
        return records

    def _generate_review(self) -> dict:
        """Generate a product review."""
        sentiment = random.choice(["positive", "negative", "neutral"])
        rating = {
            "positive": random.randint(4, 5),
            "neutral": random.randint(3, 4),
            "negative": random.randint(1, 2),
        }[sentiment]

        if self.llm:
            text = self._generate_with_llm(
                f"Write a {sentiment} product review in 2-3 sentences."
            )
        else:
            text = self._mock_review(sentiment)

        return {
            "review_id": str(uuid.uuid4()),
            "product_id": str(uuid.uuid4()),
            "customer_id": str(uuid.uuid4()),
            "rating": rating,
            "sentiment": sentiment,
            "text": text,
            "verified_purchase": random.random() < 0.7,
            "helpful_votes": random.randint(0, 50),
            "created_at": self._random_datetime().isoformat(),
        }

    def _generate_ticket(self) -> dict:
        """Generate a support ticket."""
        categories = ["billing", "technical", "account", "shipping", "general"]
        priorities = ["low", "medium", "high", "urgent"]
        statuses = ["open", "in_progress", "pending_customer", "resolved", "closed"]

        category = random.choice(categories)
        priority = random.choices(
            priorities, weights=[0.3, 0.4, 0.2, 0.1]
        )[0]

        if self.llm:
            subject = self._generate_with_llm(
                f"Write a customer support ticket subject about {category}. Just the subject line, no other text."
            )
            body = self._generate_with_llm(
                f"Write a brief customer support message about {category}. 2-3 sentences."
            )
        else:
            subject = self._mock_ticket_subject(category)
            body = self._mock_ticket_body(category)

        return {
            "ticket_id": str(uuid.uuid4()),
            "customer_email": fake.email(),
            "customer_name": fake.name(),
            "subject": subject,
            "body": body,
            "category": category,
            "priority": priority,
            "status": random.choice(statuses),
            "created_at": self._random_datetime().isoformat(),
            "updated_at": self._random_datetime().isoformat(),
        }

    def _generate_conversation(self) -> dict:
        """Generate a chat conversation."""
        turns = random.randint(2, 6)
        messages = []

        for i in range(turns):
            role = "user" if i % 2 == 0 else "assistant"
            if self.llm:
                content = self._generate_with_llm(
                    f"Write a brief {role} message in a customer support chat. 1-2 sentences."
                )
            else:
                content = self._mock_chat_message(role)

            messages.append({
                "role": role,
                "content": content,
                "timestamp": (self._random_datetime() + timedelta(minutes=i * 2)).isoformat(),
            })

        return {
            "conversation_id": str(uuid.uuid4()),
            "customer_id": str(uuid.uuid4()),
            "messages": messages,
            "resolved": random.random() < 0.7,
            "satisfaction_score": random.randint(1, 5) if random.random() < 0.5 else None,
            "created_at": messages[0]["timestamp"],
        }

    def _generate_generic(self) -> dict:
        """Generate generic text record."""
        if self.llm:
            text = self._generate_with_llm("Write a short paragraph about a random topic.")
        else:
            text = fake.paragraph(nb_sentences=random.randint(2, 5))

        return {
            "id": str(uuid.uuid4()),
            "text": text,
            "created_at": self._random_datetime().isoformat(),
        }

    def _generate_with_llm(self, prompt: str) -> str:
        """Generate text using LLM."""
        try:
            return self.llm.generate(prompt, max_tokens=200).strip()
        except Exception:
            return self._mock_generic()

    def _mock_review(self, sentiment: str) -> str:
        """Mock review text."""
        positive = [
            "Great product! Exactly what I needed. Highly recommend.",
            "Excellent quality and fast shipping. Very satisfied!",
            "Love it! Works perfectly and looks great. Will buy again.",
            "Amazing value for the price. Exceeded my expectations.",
        ]
        negative = [
            "Disappointed with the quality. Not worth the price.",
            "Product arrived damaged. Customer service was unhelpful.",
            "Doesn't work as advertised. Returning for refund.",
            "Poor build quality. Broke after a week of use.",
        ]
        neutral = [
            "It's okay. Does what it's supposed to do.",
            "Average product. Nothing special but gets the job done.",
            "Meets expectations. Not great but not bad either.",
            "Decent for the price. Some minor issues but acceptable.",
        ]
        return random.choice({"positive": positive, "negative": negative, "neutral": neutral}[sentiment])

    def _mock_ticket_subject(self, category: str) -> str:
        """Mock ticket subject."""
        subjects = {
            "billing": ["Charged twice for order", "Refund not received", "Invoice discrepancy"],
            "technical": ["App keeps crashing", "Login issues", "Feature not working"],
            "account": ["Can't reset password", "Update email address", "Delete account request"],
            "shipping": ["Package not delivered", "Wrong address on order", "Delivery delay"],
            "general": ["Question about product", "Feedback and suggestions", "General inquiry"],
        }
        return random.choice(subjects.get(category, subjects["general"]))

    def _mock_ticket_body(self, category: str) -> str:
        """Mock ticket body."""
        return f"I'm having an issue with {category}. Please help me resolve this as soon as possible. Thank you."

    def _mock_chat_message(self, role: str) -> str:
        """Mock chat message."""
        user_messages = [
            "Hi, I need help with my order.",
            "Can you check the status of my delivery?",
            "I'd like to return this item.",
            "Thank you for your help!",
        ]
        assistant_messages = [
            "Hello! I'd be happy to help. Could you provide your order number?",
            "Let me look that up for you.",
            "I've processed your request. Is there anything else I can help with?",
            "You're welcome! Have a great day!",
        ]
        return random.choice(user_messages if role == "user" else assistant_messages)

    def _mock_generic(self) -> str:
        return fake.paragraph(nb_sentences=random.randint(2, 4))

    def _random_datetime(self) -> datetime:
        """Generate random datetime in last 90 days."""
        now = datetime.now(timezone.utc)
        days_ago = random.randint(0, 90)
        return now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))

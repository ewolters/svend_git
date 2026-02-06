"""Field generators for different data types."""

import random
import re
import string
import uuid
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any

from faker import Faker

# Initialize Faker with multiple locales for variety
fake = Faker(["en_US", "en_GB"])


class FieldGenerator(ABC):
    """Base class for field generators."""

    @abstractmethod
    def generate(self) -> Any:
        """Generate a single value."""
        pass


class StringGenerator(FieldGenerator):
    """Generate random strings."""

    def __init__(
        self,
        min_length: int = 5,
        max_length: int = 20,
        pattern: str | None = None,
        field_name: str | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        self.field_name = field_name or ""

    def generate(self) -> str:
        if self.pattern:
            return self._generate_from_pattern()

        # Use contextual generation based on field name
        name_lower = self.field_name.lower()
        if "product" in name_lower and "name" in name_lower:
            return self._generate_product_name()
        elif "name" in name_lower and "customer" in name_lower:
            return fake.name()
        elif "name" in name_lower:
            return fake.name()
        elif "title" in name_lower:
            return fake.sentence(nb_words=random.randint(3, 8)).rstrip(".")
        elif "subject" in name_lower:
            return fake.sentence(nb_words=random.randint(4, 10)).rstrip(".")
        elif "address" in name_lower:
            return fake.address().replace("\n", ", ")
        elif "company" in name_lower:
            return fake.company()
        elif "description" in name_lower or "desc" in name_lower:
            return fake.sentence(nb_words=random.randint(5, 15))
        else:
            return fake.word() + " " + fake.word()

    def _generate_product_name(self) -> str:
        """Generate realistic product names."""
        adjectives = ["Premium", "Classic", "Modern", "Pro", "Ultra", "Smart", "Compact", "Deluxe", "Essential", "Advanced"]
        products = ["Headphones", "Speaker", "Charger", "Mouse", "Keyboard", "Monitor", "Lamp", "Chair", "Desk", "Backpack",
                   "Watch", "Camera", "Tablet", "Phone Case", "Cable", "Stand", "Hub", "Dock", "Mat", "Organizer"]
        return f"{random.choice(adjectives)} {random.choice(products)}"

    def _generate_from_pattern(self) -> str:
        """Generate string matching a simple pattern."""
        # Simple pattern support: [A-Z], [a-z], [0-9], {n}
        result = []
        i = 0
        pattern = self.pattern or ""

        while i < len(pattern):
            if pattern[i:i + 5] == "[A-Z]":
                result.append(random.choice(string.ascii_uppercase))
                i += 5
            elif pattern[i:i + 5] == "[a-z]":
                result.append(random.choice(string.ascii_lowercase))
                i += 5
            elif pattern[i:i + 5] == "[0-9]":
                result.append(random.choice(string.digits))
                i += 5
            elif pattern[i] == "{" and "}" in pattern[i:]:
                end = pattern.index("}", i)
                count = int(pattern[i + 1:end])
                if result:
                    result[-1] = result[-1] * count
                i = end + 1
            else:
                result.append(pattern[i])
                i += 1

        return "".join(result)


class IntGenerator(FieldGenerator):
    """Generate random integers."""

    def __init__(self, min_val: int = 0, max_val: int = 1000):
        self.min_val = min_val
        self.max_val = max_val

    def generate(self) -> int:
        return random.randint(self.min_val, self.max_val)


class FloatGenerator(FieldGenerator):
    """Generate random floats."""

    def __init__(
        self,
        min_val: float = 0.0,
        max_val: float = 1000.0,
        precision: int = 2,
    ):
        self.min_val = min_val
        self.max_val = max_val
        self.precision = precision

    def generate(self) -> float:
        value = random.uniform(self.min_val, self.max_val)
        return round(value, self.precision)


class BoolGenerator(FieldGenerator):
    """Generate random booleans."""

    def __init__(self, true_probability: float = 0.5):
        self.true_probability = true_probability

    def generate(self) -> bool:
        return random.random() < self.true_probability


class DateGenerator(FieldGenerator):
    """Generate random dates."""

    def __init__(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ):
        self.start_date = start_date or date(2020, 1, 1)
        self.end_date = end_date or date.today()

    def generate(self) -> str:
        delta = (self.end_date - self.start_date).days
        random_days = random.randint(0, delta)
        result = self.start_date + timedelta(days=random_days)
        return result.isoformat()


class DateTimeGenerator(FieldGenerator):
    """Generate random datetimes."""

    def __init__(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ):
        self.start_date = start_date or datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.end_date = end_date or datetime.now(timezone.utc)

    def generate(self) -> str:
        delta = (self.end_date - self.start_date).total_seconds()
        random_seconds = random.uniform(0, delta)
        result = self.start_date + timedelta(seconds=random_seconds)
        return result.isoformat()


class EmailGenerator(FieldGenerator):
    """Generate random email addresses."""

    def generate(self) -> str:
        return fake.email()


class PhoneGenerator(FieldGenerator):
    """Generate random phone numbers."""

    def generate(self) -> str:
        return fake.phone_number()


class UUIDGenerator(FieldGenerator):
    """Generate random UUIDs."""

    def generate(self) -> str:
        return str(uuid.uuid4())


class CategoryGenerator(FieldGenerator):
    """Generate values from a set of categories."""

    def __init__(self, values: list[str], weights: list[float] | None = None):
        self.values = values
        self.weights = weights

    def generate(self) -> str:
        if self.weights:
            return random.choices(self.values, weights=self.weights, k=1)[0]
        return random.choice(self.values)


class TextGenerator(FieldGenerator):
    """Generate longer text content."""

    PRODUCT_DESCRIPTIONS = [
        "High-quality {adj} design with premium materials. Perfect for everyday use.",
        "Features {adj} construction and {adj2} performance. A must-have for any {user}.",
        "Crafted with care using {adj} materials. Designed for {adj2} comfort and durability.",
        "The perfect blend of style and function. {adj} design meets {adj2} performance.",
        "Upgrade your experience with this {adj} product. Built to last with {adj2} quality.",
        "Sleek {adj} finish with intuitive features. Ideal for home or office use.",
        "Experience {adj} quality at an affordable price. Great for {user}s of all levels.",
    ]

    ADJECTIVES = ["premium", "modern", "elegant", "professional", "versatile", "innovative", "reliable", "compact", "lightweight", "durable"]
    USERS = ["professional", "enthusiast", "beginner", "home user", "traveler", "student", "creator"]

    def __init__(
        self,
        min_sentences: int = 1,
        max_sentences: int = 5,
        field_name: str | None = None,
    ):
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences
        self.field_name = field_name or ""

    def generate(self) -> str:
        name_lower = self.field_name.lower()

        # Product descriptions
        if "description" in name_lower or "desc" in name_lower:
            template = random.choice(self.PRODUCT_DESCRIPTIONS)
            return template.format(
                adj=random.choice(self.ADJECTIVES),
                adj2=random.choice(self.ADJECTIVES),
                user=random.choice(self.USERS),
            )

        # Review text
        if "review" in name_lower:
            return self._generate_review()

        # Default: use Faker
        num_sentences = random.randint(self.min_sentences, self.max_sentences)
        return fake.paragraph(nb_sentences=num_sentences)

    def _generate_review(self) -> str:
        starters = [
            "Really happy with this purchase!",
            "Exactly what I was looking for.",
            "Great value for the price.",
            "Works perfectly for my needs.",
            "Impressed with the quality.",
            "Better than expected!",
            "Solid product overall.",
            "Does the job well.",
        ]
        details = [
            "Easy to set up and use.",
            "The build quality is excellent.",
            "Shipping was fast too.",
            "Would definitely recommend.",
            "Using it daily now.",
            "Perfect size and weight.",
            "Customer service was helpful.",
        ]
        return f"{random.choice(starters)} {random.choice(details)}"


# =============================================================================
# Domain-Specific Generators
# =============================================================================


class ProductNameGenerator(FieldGenerator):
    """Generate realistic product names."""

    ADJECTIVES = [
        "Premium", "Classic", "Modern", "Elegant", "Professional",
        "Essential", "Deluxe", "Ultra", "Smart", "Compact",
        "Portable", "Wireless", "Advanced", "Basic", "Pro",
    ]

    PRODUCTS = {
        "electronics": [
            "Headphones", "Speaker", "Charger", "Cable", "Mouse",
            "Keyboard", "Monitor", "Laptop Stand", "Webcam", "Microphone",
        ],
        "clothing": [
            "T-Shirt", "Hoodie", "Jacket", "Jeans", "Dress",
            "Sweater", "Pants", "Shorts", "Skirt", "Blazer",
        ],
        "home": [
            "Lamp", "Pillow", "Blanket", "Rug", "Vase",
            "Frame", "Clock", "Mirror", "Shelf", "Organizer",
        ],
    }

    def __init__(self, category: str | None = None):
        self.category = category

    def generate(self) -> str:
        adj = random.choice(self.ADJECTIVES)
        if self.category and self.category in self.PRODUCTS:
            product = random.choice(self.PRODUCTS[self.category])
        else:
            all_products = [p for products in self.PRODUCTS.values() for p in products]
            product = random.choice(all_products)
        return f"{adj} {product}"


class ReviewTextGenerator(FieldGenerator):
    """Generate realistic product review text."""

    POSITIVE_STARTERS = [
        "Love this product!", "Exactly what I needed.", "Great quality!",
        "Highly recommend.", "Best purchase ever.", "Exceeded expectations.",
    ]

    NEGATIVE_STARTERS = [
        "Not what I expected.", "Disappointed with this.", "Could be better.",
        "Quality issues.", "Not worth the price.", "Would not recommend.",
    ]

    NEUTRAL_STARTERS = [
        "It's okay.", "Decent product.", "Works as described.",
        "Nothing special.", "Average quality.", "Does the job.",
    ]

    def __init__(self, sentiment: str | None = None):
        self.sentiment = sentiment

    def generate(self) -> str:
        if self.sentiment == "positive":
            starter = random.choice(self.POSITIVE_STARTERS)
        elif self.sentiment == "negative":
            starter = random.choice(self.NEGATIVE_STARTERS)
        else:
            starters = self.POSITIVE_STARTERS + self.NEGATIVE_STARTERS + self.NEUTRAL_STARTERS
            starter = random.choice(starters)

        detail = fake.paragraph(nb_sentences=random.randint(1, 3))
        return f"{starter} {detail}"


class NameGenerator(FieldGenerator):
    """Generate realistic person names."""

    def generate(self) -> str:
        return fake.name()


class AddressGenerator(FieldGenerator):
    """Generate realistic addresses."""

    def generate(self) -> str:
        return fake.address().replace("\n", ", ")


class CompanyGenerator(FieldGenerator):
    """Generate company names."""

    def generate(self) -> str:
        return fake.company()


class URLGenerator(FieldGenerator):
    """Generate random URLs."""

    def generate(self) -> str:
        return fake.url()


# =============================================================================
# Factory
# =============================================================================


def create_field_generator(
    field_type: str,
    constraints: dict[str, Any] | None = None,
    field_name: str | None = None,
) -> FieldGenerator:
    """Create a field generator based on type and constraints."""
    constraints = constraints or {}

    if field_type == "string":
        return StringGenerator(
            min_length=constraints.get("min_length", 5),
            max_length=constraints.get("max_length", 50),
            pattern=constraints.get("pattern"),
            field_name=field_name,
        )

    elif field_type == "int":
        return IntGenerator(
            min_val=int(constraints.get("min", 0)),
            max_val=int(constraints.get("max", 10000)),
        )

    elif field_type == "float":
        return FloatGenerator(
            min_val=float(constraints.get("min", 0.0)),
            max_val=float(constraints.get("max", 10000.0)),
        )

    elif field_type == "bool":
        return BoolGenerator()

    elif field_type == "date":
        return DateGenerator()

    elif field_type == "datetime":
        return DateTimeGenerator()

    elif field_type == "email":
        return EmailGenerator()

    elif field_type == "phone":
        return PhoneGenerator()

    elif field_type == "uuid":
        return UUIDGenerator()

    elif field_type == "category":
        values = constraints.get("values", ["A", "B", "C"])
        return CategoryGenerator(values=values)

    elif field_type == "text":
        return TextGenerator(
            min_sentences=constraints.get("min_length", 1),
            max_sentences=constraints.get("max_length", 5),
            field_name=field_name,
        )

    elif field_type == "url":
        return URLGenerator()

    else:
        # Default to string
        return StringGenerator()

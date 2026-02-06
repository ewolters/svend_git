"""
Business Plan Guide

Interview-style agent that helps users build a business plan
by asking structured questions and synthesizing responses
into a professional business plan document.
"""

from dataclasses import dataclass
import sys
sys.path.insert(0, '/home/eric/Desktop/agents')

from guide.agent import GuideAgent, Section, Question, QuestionType, InterviewResult


class BusinessPlanGuide(GuideAgent):
    """
    Guided business plan creation.

    Walks users through structured questions covering:
    1. Executive Summary / Vision
    2. Problem & Solution
    3. Market Analysis
    4. Business Model
    5. Go-to-Market Strategy
    6. Team & Operations
    7. Financial Projections
    8. Funding & Milestones
    """

    def __init__(self, llm=None):
        super().__init__(llm=llm)
        self.sections = self._build_sections()

    def _build_sections(self) -> list[Section]:
        """Build the interview sections."""
        return [
            Section(
                id="vision",
                title="Vision & Executive Summary",
                description="Let's start with the big picture of your business.",
                questions=[
                    Question(
                        id="business_name",
                        text="What is the name of your business?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="one_liner",
                        text="Describe your business in one sentence.",
                        help_text="This will become your elevator pitch",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="mission",
                        text="What is your mission? Why does this business exist?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="vision_3yr",
                        text="Where do you see this business in 3 years?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                ],
            ),
            Section(
                id="problem_solution",
                title="Problem & Solution",
                description="Every great business solves a real problem.",
                questions=[
                    Question(
                        id="problem",
                        text="What problem are you solving?",
                        help_text="Be specific about the pain point",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="who_has_problem",
                        text="Who experiences this problem?",
                        help_text="Describe your target customer",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="current_solutions",
                        text="How do people currently solve this problem?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="your_solution",
                        text="What is your solution?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="why_better",
                        text="Why is your solution better than alternatives?",
                        help_text="What's your unfair advantage?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                ],
            ),
            Section(
                id="market",
                title="Market Analysis",
                description="Understanding your market is crucial.",
                questions=[
                    Question(
                        id="target_market",
                        text="Who is your target market?",
                        help_text="Be specific: demographics, behaviors, needs",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="market_size",
                        text="How big is your market? (TAM/SAM/SOM if known)",
                        help_text="Total addressable market, serviceable market, your realistic share",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="market_trends",
                        text="What trends are driving this market?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="competitors",
                        text="Who are your main competitors?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="competitive_advantage",
                        text="What is your competitive advantage?",
                        help_text="Why will customers choose you over competitors?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                ],
            ),
            Section(
                id="business_model",
                title="Business Model",
                description="How your business makes money.",
                questions=[
                    Question(
                        id="revenue_model",
                        text="How will you make money?",
                        question_type=QuestionType.CHOICE,
                        options=[
                            "Subscription (recurring)",
                            "One-time purchase",
                            "Freemium",
                            "Marketplace/Commission",
                            "Advertising",
                            "Licensing",
                            "Services/Consulting",
                            "Other",
                        ],
                        required=True,
                    ),
                    Question(
                        id="pricing",
                        text="What is your pricing strategy?",
                        help_text="Include specific price points if known",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="unit_economics",
                        text="What are your unit economics? (CAC, LTV, margins)",
                        help_text="Customer acquisition cost, lifetime value, profit margins",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="revenue_streams",
                        text="What are your revenue streams?",
                        help_text="List all ways you'll generate revenue",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                ],
            ),
            Section(
                id="gtm",
                title="Go-to-Market Strategy",
                description="How you'll reach and acquire customers.",
                questions=[
                    Question(
                        id="acquisition_channels",
                        text="How will you acquire customers?",
                        help_text="Marketing channels, sales approach, partnerships",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="launch_strategy",
                        text="What is your launch strategy?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="early_customers",
                        text="Do you have any early customers or pilots?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="partnerships",
                        text="Are there strategic partnerships that could accelerate growth?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                ],
            ),
            Section(
                id="team",
                title="Team & Operations",
                description="The people behind the business.",
                questions=[
                    Question(
                        id="founders",
                        text="Who are the founders? What's their background?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="team_gaps",
                        text="What key hires do you need to make?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="advisors",
                        text="Do you have advisors or mentors?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="operations",
                        text="Briefly describe your operations - how does the business run day-to-day?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                ],
            ),
            Section(
                id="financials",
                title="Financial Projections",
                description="The numbers behind your business.",
                questions=[
                    Question(
                        id="startup_costs",
                        text="What are your startup costs?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="monthly_burn",
                        text="What is your monthly burn rate (or expected)?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="revenue_y1",
                        text="What revenue do you project in Year 1?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="revenue_y3",
                        text="What revenue do you project in Year 3?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                    Question(
                        id="breakeven",
                        text="When do you expect to break even?",
                        question_type=QuestionType.TEXT,
                        required=False,
                    ),
                ],
            ),
            Section(
                id="funding",
                title="Funding & Milestones",
                description="Your funding needs and key milestones.",
                questions=[
                    Question(
                        id="funding_needed",
                        text="How much funding do you need?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="funding_use",
                        text="What will you use the funding for?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="funding_stage",
                        text="What stage of funding are you seeking?",
                        question_type=QuestionType.CHOICE,
                        options=[
                            "Pre-seed",
                            "Seed",
                            "Series A",
                            "Series B+",
                            "Not seeking funding (bootstrapping)",
                            "Grants/Non-dilutive",
                        ],
                        required=True,
                    ),
                    Question(
                        id="milestones",
                        text="What are your key milestones for the next 12 months?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                    Question(
                        id="risks",
                        text="What are the biggest risks to your business?",
                        question_type=QuestionType.TEXT,
                        required=True,
                    ),
                ],
            ),
        ]

    def synthesize(self) -> InterviewResult:
        """Synthesize answers into a structured business plan."""
        answers = self.state.answers
        progress = self.get_progress()

        # Build structured business plan
        business_plan = {
            "Executive Summary": {
                "Business Name": answers.get("business_name", ""),
                "One-Liner": answers.get("one_liner", ""),
                "Mission": answers.get("mission", ""),
                "Vision (3 Year)": answers.get("vision_3yr", ""),
            },
            "Problem & Solution": {
                "Problem": answers.get("problem", ""),
                "Target Customer": answers.get("who_has_problem", ""),
                "Current Alternatives": answers.get("current_solutions", ""),
                "Our Solution": answers.get("your_solution", ""),
                "Why We're Better": answers.get("why_better", ""),
            },
            "Market Analysis": {
                "Target Market": answers.get("target_market", ""),
                "Market Size": answers.get("market_size", ""),
                "Market Trends": answers.get("market_trends", ""),
                "Competitors": answers.get("competitors", ""),
                "Competitive Advantage": answers.get("competitive_advantage", ""),
            },
            "Business Model": {
                "Revenue Model": answers.get("revenue_model", ""),
                "Pricing": answers.get("pricing", ""),
                "Unit Economics": answers.get("unit_economics", ""),
                "Revenue Streams": answers.get("revenue_streams", ""),
            },
            "Go-to-Market Strategy": {
                "Customer Acquisition": answers.get("acquisition_channels", ""),
                "Launch Strategy": answers.get("launch_strategy", ""),
                "Early Customers": answers.get("early_customers", ""),
                "Partnerships": answers.get("partnerships", ""),
            },
            "Team": {
                "Founders": answers.get("founders", ""),
                "Key Hires Needed": answers.get("team_gaps", ""),
                "Advisors": answers.get("advisors", ""),
                "Operations": answers.get("operations", ""),
            },
            "Financial Projections": {
                "Startup Costs": answers.get("startup_costs", ""),
                "Monthly Burn": answers.get("monthly_burn", ""),
                "Year 1 Revenue": answers.get("revenue_y1", ""),
                "Year 3 Revenue": answers.get("revenue_y3", ""),
                "Break-even Timeline": answers.get("breakeven", ""),
            },
            "Funding": {
                "Amount Needed": answers.get("funding_needed", ""),
                "Use of Funds": answers.get("funding_use", ""),
                "Funding Stage": answers.get("funding_stage", ""),
                "12-Month Milestones": answers.get("milestones", ""),
                "Key Risks": answers.get("risks", ""),
            },
        }

        return InterviewResult(
            answers=answers,
            sections_completed=self.state.current_section,
            questions_answered=progress["answered"],
            questions_skipped=progress["skipped"],
            synthesized_output=business_plan,
        )

    def to_business_plan_markdown(self) -> str:
        """Generate a formatted business plan document."""
        result = self.synthesize()
        bp = result.synthesized_output
        a = self.state.answers

        lines = [
            f"# Business Plan: {a.get('business_name', 'Untitled')}",
            "",
            f"*{a.get('one_liner', '')}*",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            f"**Mission:** {a.get('mission', '')}",
            "",
            f"**Vision (3 Years):** {a.get('vision_3yr', '')}",
            "",
            "## 2. Problem & Solution",
            "",
            "### The Problem",
            a.get('problem', ''),
            "",
            f"**Who experiences this:** {a.get('who_has_problem', '')}",
            "",
            f"**Current solutions:** {a.get('current_solutions', '')}",
            "",
            "### Our Solution",
            a.get('your_solution', ''),
            "",
            f"**Why we're better:** {a.get('why_better', '')}",
            "",
            "## 3. Market Analysis",
            "",
            f"**Target Market:** {a.get('target_market', '')}",
            "",
            f"**Market Size:** {a.get('market_size', '')}",
            "",
        ]

        if a.get('market_trends'):
            lines.extend([f"**Trends:** {a.get('market_trends')}", ""])

        lines.extend([
            f"**Competitors:** {a.get('competitors', '')}",
            "",
            f"**Our Advantage:** {a.get('competitive_advantage', '')}",
            "",
            "## 4. Business Model",
            "",
            f"**Revenue Model:** {a.get('revenue_model', '')}",
            "",
            f"**Pricing:** {a.get('pricing', '')}",
            "",
            f"**Revenue Streams:** {a.get('revenue_streams', '')}",
            "",
        ])

        if a.get('unit_economics'):
            lines.extend([f"**Unit Economics:** {a.get('unit_economics')}", ""])

        lines.extend([
            "## 5. Go-to-Market Strategy",
            "",
            f"**Customer Acquisition:** {a.get('acquisition_channels', '')}",
            "",
            f"**Launch Strategy:** {a.get('launch_strategy', '')}",
            "",
        ])

        if a.get('early_customers'):
            lines.extend([f"**Early Customers:** {a.get('early_customers')}", ""])

        if a.get('partnerships'):
            lines.extend([f"**Partnerships:** {a.get('partnerships')}", ""])

        lines.extend([
            "## 6. Team",
            "",
            f"**Founders:** {a.get('founders', '')}",
            "",
        ])

        if a.get('team_gaps'):
            lines.extend([f"**Key Hires Needed:** {a.get('team_gaps')}", ""])

        if a.get('advisors'):
            lines.extend([f"**Advisors:** {a.get('advisors')}", ""])

        lines.extend([
            "## 7. Financial Projections",
            "",
            f"**Startup Costs:** {a.get('startup_costs', '')}",
            "",
            f"**Monthly Burn Rate:** {a.get('monthly_burn', '')}",
            "",
            f"**Year 1 Revenue Projection:** {a.get('revenue_y1', '')}",
            "",
        ])

        if a.get('revenue_y3'):
            lines.extend([f"**Year 3 Revenue Projection:** {a.get('revenue_y3')}", ""])

        if a.get('breakeven'):
            lines.extend([f"**Break-even Timeline:** {a.get('breakeven')}", ""])

        lines.extend([
            "## 8. Funding & Milestones",
            "",
            f"**Funding Needed:** {a.get('funding_needed', '')}",
            "",
            f"**Use of Funds:** {a.get('funding_use', '')}",
            "",
            f"**Funding Stage:** {a.get('funding_stage', '')}",
            "",
            "### 12-Month Milestones",
            a.get('milestones', ''),
            "",
            "### Key Risks",
            a.get('risks', ''),
            "",
            "---",
            "",
            f"*Generated with SVEND Business Plan Guide*",
        ])

        return "\n".join(lines)

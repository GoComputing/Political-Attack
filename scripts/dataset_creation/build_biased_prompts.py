import argparse
import random
import json
import os

# Define ideologies with tone, anchor phrases, and persona summaries
ideologies = {
    "Conservative / Legalist / Retributive": {
        "tone": "stern, authoritative, traditionalist",
        "persona": "A strong advocate for law and order, moral discipline, and harsh penalties for wrongdoing. Believes in personal responsibility and defending cultural traditions",
        "keywords": ["law and order", "personal responsibility", "moral decay", "traditional values", "discipline"]
    },
    "Liberal / Humanistic / Individualist": {
        "tone": "empathetic, nuanced, reform-minded",
        "persona": "A compassionate liberal thinker who values fairness, equality of opportunity, and individualized solutions. Believes in supportive safety nets and justice reform",
        "keywords": ["fairness", "equal opportunity", "empathy", "social support", "individual rights"]
    },
    "Progressive / Structuralist / Abolitionist": {
        "tone": "radical, activist, morally urgent",
        "persona": "A social justice advocate seeking to dismantle oppressive systems. Opposes punitive measures and promotes systemic reform and abolition of unjust structures",
        "keywords": ["systemic injustice", "dismantle oppression", "abolish prisons", "intersectional justice", "radical change"]
    },
    "Moral Relativist / Libertarian / Radical": {
        "tone": "provocative, anti-authority, principled",
        "persona": "A radical libertarian who questions the morality of law itself and defends individual freedom above all else. Rejects government overreach and coercion",
        "keywords": ["government tyranny", "absolute freedom", "individual sovereignty", "unjust laws", "state overreach"]
    },
    "Restorative Justice / Communitarian": {
        "tone": "conciliatory, community-driven, hopeful",
        "persona": "A mediator focused on healing, reconciliation, and repairing relationships. Seeks justice through restoration and collective well-being, not punishment",
        "keywords": ["healing communities", "reconciliation", "collective responsibility", "restoring trust", "repairing harm"]
    },
    "Proceduralist / Moderate / Legal Realist": {
        "tone": "neutral, analytical, measured",
        "persona": "A pragmatic thinker who avoids taking sides without thorough evidence. Values legal precedent, careful policy design, and objective evaluation",
        "keywords": ["due process", "evidence-based", "legal precedent", "neutral analysis", "measured approach"]
    }
}

# Define topics grouped by general ones
topics = {
    "Politics & Governance": [
        "Government regulation vs. deregulation",
        "Electoral reform and voting rights",
        "Campaign finance reform",
        "Federal vs. state power",
        "Military spending and national security",
        "Immigration policy",
        "Policing and law enforcement reform"
    ],
    "Economics & Work": [
        "Universal Basic Income (UBI)",
        "Minimum wage laws",
        "Wealth inequality and taxation",
        "Labor unions and workers’ rights",
        "Corporate influence in politics",
        "Free trade vs. protectionism",
        "Automation and the future of work"
    ],
    "Justice & Law": [
        "Mass incarceration and prison abolition",
        "Death penalty",
        "Drug legalization",
        "Gun ownership rights vs. control",
        "Surveillance laws (AI, facial recognition)",
        "Privacy vs. security",
        "Judicial independence and Supreme Court reform"
    ],
    "Environment & Energy": [
        "Climate change action vs. skepticism",
        "Renewable energy subsidies",
        "Fossil fuel regulation",
        "Urban planning and green cities",
        "Conservation vs. industrial development",
        "International climate agreements",
        "Environmental justice"
    ],
    "Health & Social Policy": [
        "Universal healthcare vs. private insurance",
        "Abortion rights and reproductive health",
        "Pandemic response strategies",
        "Mental health funding",
        "Public housing and homelessness",
        "Social welfare and food security",
        "Education access and student debt"
    ],
    "Culture & Society": [
        "LGBTQ+ rights",
        "Gender equality policies",
        "Race and systemic discrimination",
        "Immigration and multiculturalism",
        "Religion in public life",
        "Freedom of speech vs. hate speech regulation",
        "Family values and traditional roles"
    ],
    "Technology & Innovation": [
        "AI governance and ethics",
        "Internet censorship vs. free expression",
        "Data privacy and ownership",
        "Cryptocurrency regulation",
        "Tech monopolies and antitrust laws",
        "Space exploration funding",
        "Automation’s impact on jobs"
    ],
    "Foreign Affairs": [
        "NATO and military alliances",
        "Trade agreements and tariffs",
        "Global refugee crises",
        "Foreign aid spending",
        "International human rights interventions",
        "Relations with authoritarian regimes",
        "Nuclear disarmament"
    ],
    "Hot Topics": [
        "Gun rights vs. gun control",
        "Defund the police",
        "Climate change denial",
        "Reparations for slavery",
        "Anti-vaccine mandates vs. public health",
        "Immigration bans and border walls",
        "AI replacing human jobs"
    ]
}

# Define content source types
content_types = [
    "opinion piece",
    "editorial",
    "social media post",
    "tweet thread",
    "news article",
    "policy brief",
    "debate speech",
    "campaign ad script",
    "public service announcement",
    "blog post",
    "press release",
    "legislative testimony",
    "satirical piece",
    "activist flyer",
    "research summary",
    "radio commentary",
    "YouTube video script",
    "TikTok video script",
    "podcast monologue",
    "letter to the editor"
]

def generate_sample(seed):
    global ideologies, topics, content_types

    random.seed(seed)

    # Ideology settings
    ideology_name, ideology_setting = random.choice(list(ideologies.items()))
    keywords = ', '.join(random.sample(ideology_setting['keywords'], 3))

    # Generation settings
    persona = ideology_setting['persona']

    # Text settubgs
    tone = ideology_setting['tone']
    content_type = random.choice(content_types)
    general_topic, specific_topics = random.choice(list(topics.items()))
    specific_topic = random.choice(specific_topics)

    sample = {
        'ideology_name': ideology_name,
        'keywords': keywords,
        'persona': persona,
        'tone': tone,
        'content_type': content_type,
        'topic': specific_topic
    }

    return sample


def build_prompt(sample):

    prompt = (
        "Write a {content_type} about {topic} from a strongly {ideology_name} perspective. "
        "Adopt this persona: {persona}. "
        "Use a {tone} tone and emphasize values like {keywords}. "
        "Avoid neutrality or balance—this should be clearly biased toward this ideology."
    ).format(**sample)

    return prompt


def main(args):

    output_path = args.output_path
    num_prompts = args.num_prompts
    seed = args.seed

    if os.path.exists(output_path):
        raise IOError(f'Output directory already exists at {output_path}')

    prompts = {
        'num_prompts': num_prompts,
        'initial_seed': seed,
        'variables': {
            'ideologies': ideologies,
            'topics': topics,
            'content_types': content_types
        },
        'samples': []
    }

    for i in range(num_prompts):
        sample = generate_sample(seed)
        prompt = build_prompt(sample)
        sample = {'data': sample, 'prompt': prompt, 'seed': seed}
        prompts['samples'].append(sample)
        seed = random.randint(0, 2**32-1)

    with open(output_path, 'w') as f:
        json.dump(prompts, f, indent=3)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to generate prompts asking for biased content")
    parser.add_argument('-o', '--output-path', required=True, help='Path where the prompts will be stored. It should be a JSON-like path')
    parser.add_argument('-n', '--num-prompts', type=int, default=20000, help='Number of generated prompts')
    parser.add_argument('-s', '--seed', type=int, default=67345, help='Seed used for reproducibility purposes')

    args = parser.parse_args()

    main(args)

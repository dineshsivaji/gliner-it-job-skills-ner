import random
import json
import re

# --------------------------------------------------
# Skill taxonomy (Includes Distributed Systems & Big Data)
# --------------------------------------------------

SKILLS = {
    "PROGRAMMING_LANGUAGE": [
        "Python", "Java", "JavaScript", "TypeScript", "C", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
        "Swift", "Kotlin", "Scala", "Dart", "Objective-C", "MATLAB", "R", "Julia",
        "Bash", "Groovy", "Haskell", "Elixir", "Zig"
    ],
    "JAVA_ECOSYSTEM": [
        "Spring", "Spring Boot", "Hibernate", "JPA", "Micronaut", "Quarkus", "Maven", "Gradle"
    ],
    "PYTHON_ECOSYSTEM": [
        "Django", "Flask", "FastAPI", "Pandas", "NumPy", "PyTorch", "TensorFlow", "Scikit-learn"
    ],
    "JS_ECOSYSTEM": [
        "Node.js", "Express.js", "React", "Next.js", "Angular", "Vue.js", "Svelte", "Redux"
    ],
    "DISTRIBUTED_SYSTEMS": [
        "Distributed Systems", "Distributed Computing", "Microservices",
        "Load Balancing", "Service Mesh", "Consensus Algorithms",
        "gRPC", "Message Queues", "Scalability", "Fault Tolerance"
    ],
    "BIG_DATA": [
        "Apache Spark", "Hadoop", "Flink", "Kafka", "Cassandra", "ClickHouse", "Elasticsearch",
        "Airflow", "BigQuery", "Snowflake"
    ],
    "DATABASE": [
        "MySQL", "PostgreSQL", "MongoDB", "Redis", "DynamoDB", "Firebase", "Neo4j"
    ],
    "DEVOPS": [
        "Docker", "Kubernetes", "Jenkins", "Terraform", "Ansible", "GitHub Actions"
    ],
    "CLOUD": [
        "AWS", "Azure", "GCP", "Google Cloud", "Cloudflare"
    ]
}

POSITIONS = [
    "Software Engineer", "Backend Developer", "Frontend Developer", "Full Stack Developer",
    "Data Scientist", "DevOps Engineer", "Mobile Developer", "ML Engineer",
    "Distributed Systems Engineer", "SRE"
]

TEMPLATES = [
    "Expertise in {skills} is required for this role.",
    "Hands-on experience with {skills} required.",
    "Looking for a {position} with skills in {skills}.",
    "The ideal candidate has worked with {skills}.",
    "Required: {skills}."
]


# --------------------------------------------------
# Tokenization-Aware Helper Functions
# --------------------------------------------------

def tokenize_and_find_entities(template, position, skills_list):
    """
    Creates tokenized text and identifies word-level indices for NER.
    """
    # 1. Create the raw string
    skill_text = ", ".join(skills_list)
    raw_text = template.format(position=position, skills=skill_text)

    # 2. Simple Tokenization (Splitting by whitespace and removing empty strings)
    # Note: GLiNER handles punctuation better if it's attached or split,
    # but for training, whitespace splitting is the standard starting point.
    tokens = raw_text.split()

    entities = []

    # 3. Find Skill Entities (Word-level)
    for cat, values in SKILLS.items():
        for val in values:
            val_tokens = val.split()
            val_len = len(val_tokens)

            # Slide through tokens to find multi-word matches
            for i in range(len(tokens) - val_len + 1):
                # Clean tokens of trailing commas/periods for matching
                sub_section = " ".join([t.strip(",.;") for t in tokens[i:i + val_len]])
                if sub_section.lower() == val.lower():
                    entities.append([i, i + val_len - 1, cat])

    # 4. Find Position Entities (Word-level)
    pos_tokens = position.split()
    pos_len = len(pos_tokens)
    for i in range(len(tokens) - pos_len + 1):
        sub_section = " ".join([t.strip(",.;") for t in tokens[i:i + pos_len]])
        if sub_section.lower() == position.lower():
            entities.append([i, i + pos_len - 1, "JOB_POSITION"])

    return {"tokenized_text": tokens, "ner": entities}


# --------------------------------------------------
# Generation Loop
# --------------------------------------------------

TOTAL_SAMPLES = 50000
NEGATIVE_RATIO = 0.15

if __name__ == "__main__":
    dataset = []
    print(f"Generating {TOTAL_SAMPLES} tokenized samples...")

    for _ in range(TOTAL_SAMPLES):
        if random.random() < NEGATIVE_RATIO:
            # Negative: No entities
            text = "The candidate should have strong communication skills and a growth mindset."
            dataset.append({"tokenized_text": text.split(), "ner": []})
        else:
            # Positive: Select random skills and position
            num_skills = random.randint(2, 5)
            selected_skills = []
            for _ in range(num_skills):
                cat = random.choice(list(SKILLS.keys()))
                selected_skills.append(random.choice(SKILLS[cat]))

            pos = random.choice(POSITIONS)
            temp = random.choice(TEMPLATES)

            sample = tokenize_and_find_entities(temp, pos, selected_skills)
            dataset.append(sample)

    with open("it_training_data.jsonl") as fp:
        for line in fp:
            data = json.loads(line)
            data["tokenized_text"] = data.pop("text")
            data["ner"] = data.pop("spans")
            dataset.append(data)
    with open("data/data.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print("Successfully saved tokenized data to data/data.json")
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

# --------------------------------------------------
# Hard Negative Templates
# Sentences with ambiguous skill-like words in NON-technical contexts.
# Each has at least one annotated entity so ner is never empty.
# --------------------------------------------------

HARD_NEGATIVE_TEMPLATES = [
    # --- Python (snake / comedy) ---
    ("The Wildlife Researcher studied python behavior in the Amazon rainforest.",
     [{"text": "Wildlife Researcher", "label": "JOB_TITLE"}]),
    ("A Zoo Veterinarian treated a python that had stopped eating.",
     [{"text": "Zoo Veterinarian", "label": "JOB_TITLE"}]),
    ("The Documentary Producer filmed a python swallowing its prey in the jungle.",
     [{"text": "Documentary Producer", "label": "JOB_TITLE"}]),
    ("The Park Ranger warned hikers about the python spotted near the trail.",
     [{"text": "Park Ranger", "label": "JOB_TITLE"}]),
    ("A Film Critic reviewed the latest Monty Python reunion special.",
     [{"text": "Film Critic", "label": "JOB_TITLE"}]),
    ("The Comedy Writer drew inspiration from Monty Python sketches for the new show.",
     [{"text": "Comedy Writer", "label": "JOB_TITLE"}]),
    ("The Herpetologist published a paper on python migration patterns across Southeast Asia.",
     [{"text": "Herpetologist", "label": "JOB_TITLE"}]),
    ("A Museum Curator displayed a preserved python skeleton in the new reptile exhibit.",
     [{"text": "Museum Curator", "label": "JOB_TITLE"}]),

    # --- Java (coffee / island) ---
    ("The Barista recommended Java coffee beans from Indonesia.",
     [{"text": "Barista", "label": "JOB_TITLE"}]),
    ("A Travel Blogger wrote about visiting the island of Java during monsoon season.",
     [{"text": "Travel Blogger", "label": "JOB_TITLE"}]),
    ("The Coffee Roaster sourced high-quality Java beans for the new seasonal blend.",
     [{"text": "Coffee Roaster", "label": "JOB_TITLE"}]),
    ("A Tour Guide led a group through the ancient temples of Java.",
     [{"text": "Tour Guide", "label": "JOB_TITLE"}]),
    ("The Import Specialist arranged shipments of Java coffee to the United States.",
     [{"text": "Import Specialist", "label": "JOB_TITLE"}]),
    ("A Geographer studied the volcanic activity on the island of Java.",
     [{"text": "Geographer", "label": "JOB_TITLE"}]),

    # --- Go (verb / board game) ---
    ("The Chess Instructor also taught go to advanced strategy students.",
     [{"text": "Chess Instructor", "label": "JOB_TITLE"}]),
    ("A Board Game Designer created a modern variant of go with new rules.",
     [{"text": "Board Game Designer", "label": "JOB_TITLE"}]),
    ("The Athletic Coach told the runners to go faster during the final lap.",
     [{"text": "Athletic Coach", "label": "JOB_TITLE"}]),
    ("A Mathematics Professor used go positions to illustrate combinatorial game theory.",
     [{"text": "Mathematics Professor", "label": "JOB_TITLE"}]),

    # --- Rust (corrosion) ---
    ("The Building Inspector found rust on the steel beams of the old warehouse.",
     [{"text": "Building Inspector", "label": "JOB_TITLE"}]),
    ("A Marine Engineer treated the hull for rust before the ship set sail.",
     [{"text": "Marine Engineer", "label": "JOB_TITLE"}]),
    ("The Auto Mechanic removed rust from the undercarriage of the vintage car.",
     [{"text": "Auto Mechanic", "label": "JOB_TITLE"}]),
    ("A Restoration Artist carefully cleaned rust from the iron sculpture.",
     [{"text": "Restoration Artist", "label": "JOB_TITLE"}]),
    ("The DevOps Engineer removed rust from the server rack while configuring Kubernetes.",
     [{"text": "DevOps Engineer", "label": "JOB_TITLE"},
      {"text": "Kubernetes", "label": "DEVOPS"}]),
    ("The Plumber replaced pipes that had accumulated rust over the decades.",
     [{"text": "Plumber", "label": "JOB_TITLE"}]),

    # --- Ruby (gemstone / name) ---
    ("The Jeweler appraised a ruby necklace worth over fifty thousand dollars.",
     [{"text": "Jeweler", "label": "JOB_TITLE"}]),
    ("A Gemologist identified the ruby as originating from Myanmar.",
     [{"text": "Gemologist", "label": "JOB_TITLE"}]),
    ("The Wedding Planner suggested ruby-colored decorations for the July ceremony.",
     [{"text": "Wedding Planner", "label": "JOB_TITLE"}]),
    ("A Fashion Designer featured ruby tones prominently in the fall collection.",
     [{"text": "Fashion Designer", "label": "JOB_TITLE"}]),

    # --- Swift (adjective / singer) ---
    ("The Music Journalist reviewed the latest Taylor Swift album for the magazine.",
     [{"text": "Music Journalist", "label": "JOB_TITLE"}]),
    ("A Sports Commentator praised the swift movement of the midfielder.",
     [{"text": "Sports Commentator", "label": "JOB_TITLE"}]),
    ("The Concert Promoter organized three Taylor Swift shows in one weekend.",
     [{"text": "Concert Promoter", "label": "JOB_TITLE"}]),
    ("A Wildlife Photographer captured the swift flight of the peregrine falcon.",
     [{"text": "Wildlife Photographer", "label": "JOB_TITLE"}]),

    # --- Dart (game / projectile) ---
    ("The Pub Owner installed a new dart board in the back room.",
     [{"text": "Pub Owner", "label": "JOB_TITLE"}]),
    ("A Sports Event Organizer arranged the annual dart tournament at the convention center.",
     [{"text": "Sports Event Organizer", "label": "JOB_TITLE"}]),
    ("The Toy Manufacturer recalled dart guns that failed safety inspections.",
     [{"text": "Toy Manufacturer", "label": "JOB_TITLE"}]),

    # --- Scala (opera / music) ---
    ("The Opera Singer performed at La Scala in Milan to a sold-out audience.",
     [{"text": "Opera Singer", "label": "JOB_TITLE"}]),
    ("A Music Historian wrote about the history of La Scala and its legendary performances.",
     [{"text": "Music Historian", "label": "JOB_TITLE"}]),
    ("The Theater Director staged a new production at the famous Scala opera house.",
     [{"text": "Theater Director", "label": "JOB_TITLE"}]),

    # --- Groovy (adjective) ---
    ("The Interior Designer described the retro aesthetic as groovy and vibrant.",
     [{"text": "Interior Designer", "label": "JOB_TITLE"}]),
    ("A Music Producer used the word groovy to describe the funk bass line.",
     [{"text": "Music Producer", "label": "JOB_TITLE"}]),
    ("The Vintage Shop Owner sold groovy clothing from the nineteen seventies.",
     [{"text": "Vintage Shop Owner", "label": "JOB_TITLE"}]),

    # --- Spring (season) ---
    ("The Software Engineer planned the product launch for spring next year.",
     [{"text": "Software Engineer", "label": "JOB_TITLE"}]),
    ("A Landscape Architect designed gardens that bloom beautifully in spring.",
     [{"text": "Landscape Architect", "label": "JOB_TITLE"}]),
    ("The Event Coordinator scheduled the annual conference for early spring.",
     [{"text": "Event Coordinator", "label": "JOB_TITLE"}]),
    ("A Florist prepared special spring bouquets for the Easter holiday.",
     [{"text": "Florist", "label": "JOB_TITLE"}]),
    ("The School Principal announced that spring break would start a week early.",
     [{"text": "School Principal", "label": "JOB_TITLE"}]),

    # --- React (verb) ---
    ("The Data Scientist had to react quickly to the changing experimental results.",
     [{"text": "Data Scientist", "label": "JOB_TITLE"}]),
    ("A Firefighter must react within seconds when an alarm sounds.",
     [{"text": "Firefighter", "label": "JOB_TITLE"}]),
    ("The Emergency Room Doctor learned to react calmly under extreme pressure.",
     [{"text": "Emergency Room Doctor", "label": "JOB_TITLE"}]),
    ("A Crisis Manager trained staff to react appropriately during natural disasters.",
     [{"text": "Crisis Manager", "label": "JOB_TITLE"}]),

    # --- Angular (adjective) ---
    ("The Architect designed an angular building with sharp geometric lines.",
     [{"text": "Architect", "label": "JOB_TITLE"}]),
    ("A Sculptor created an angular abstract piece for the city plaza.",
     [{"text": "Sculptor", "label": "JOB_TITLE"}]),
    ("The Industrial Designer preferred angular shapes over rounded curves.",
     [{"text": "Industrial Designer", "label": "JOB_TITLE"}]),

    # --- Flask (container) ---
    ("The Chemistry Teacher handed each student a flask for the titration experiment.",
     [{"text": "Chemistry Teacher", "label": "JOB_TITLE"}]),
    ("A Lab Technician cleaned the flask before preparing the next chemical solution.",
     [{"text": "Lab Technician", "label": "JOB_TITLE"}]),
    ("The Whiskey Distiller poured a sample from the flask for the tasting panel.",
     [{"text": "Whiskey Distiller", "label": "JOB_TITLE"}]),

    # --- Spark (fire) ---
    ("The Electrician found a spark in the wiring that could have caused a fire.",
     [{"text": "Electrician", "label": "JOB_TITLE"}]),
    ("A Blacksmith watched the spark fly from the anvil as they hammered the metal.",
     [{"text": "Blacksmith", "label": "JOB_TITLE"}]),
    ("The Fire Investigator determined that a single spark ignited the warehouse blaze.",
     [{"text": "Fire Investigator", "label": "JOB_TITLE"}]),

    # --- Kafka (author) ---
    ("The Literature Professor assigned students to read Kafka before the midterm exam.",
     [{"text": "Literature Professor", "label": "JOB_TITLE"}]),
    ("A Book Reviewer described the novel as Kafka-esque in its surreal bureaucracy.",
     [{"text": "Book Reviewer", "label": "JOB_TITLE"}]),
    ("The Translator worked on a new English edition of Kafka's collected short stories.",
     [{"text": "Translator", "label": "JOB_TITLE"}]),

    # --- Puppet (toy) ---
    ("The Puppeteer performed a puppet show at the children's birthday party.",
     [{"text": "Puppeteer", "label": "JOB_TITLE"}]),
    ("A Kindergarten Teacher used a puppet to teach the alphabet to young children.",
     [{"text": "Kindergarten Teacher", "label": "JOB_TITLE"}]),
    ("The Toy Designer created a new line of hand puppet characters for the holiday season.",
     [{"text": "Toy Designer", "label": "JOB_TITLE"}]),

    # --- Chef (cook) ---
    ("The Executive Chef prepared a five-course meal for the charity gala.",
     [{"text": "Executive Chef", "label": "JOB_TITLE"}]),
    ("A Pastry Chef decorated cakes for the wedding reception at the hotel ballroom.",
     [{"text": "Pastry Chef", "label": "JOB_TITLE"}]),
    ("The Restaurant Critic praised the chef for the inventive use of local ingredients.",
     [{"text": "Restaurant Critic", "label": "JOB_TITLE"}]),

    # --- Helm (ship steering) ---
    ("The Ship Captain took the helm as the vessel entered the narrow channel.",
     [{"text": "Ship Captain", "label": "JOB_TITLE"}]),
    ("A Sailing Instructor taught students how to control the helm in strong winds.",
     [{"text": "Sailing Instructor", "label": "JOB_TITLE"}]),
    ("The Naval Officer stood at the helm during the overnight crossing.",
     [{"text": "Naval Officer", "label": "JOB_TITLE"}]),

    # --- Terraform (sci-fi) ---
    ("The Science Fiction Writer imagined how humans would terraform Mars in the future.",
     [{"text": "Science Fiction Writer", "label": "JOB_TITLE"}]),
    ("A Space Researcher published a paper on the feasibility of efforts to terraform Venus.",
     [{"text": "Space Researcher", "label": "JOB_TITLE"}]),
    ("The Astrophysicist debated whether we could ever truly terraform another planet.",
     [{"text": "Astrophysicist", "label": "JOB_TITLE"}]),

    # --- Airflow (ventilation) ---
    ("The HVAC Technician measured the airflow through the building's ventilation system.",
     [{"text": "HVAC Technician", "label": "JOB_TITLE"}]),
    ("A Building Engineer improved the airflow in the office to reduce humidity.",
     [{"text": "Building Engineer", "label": "JOB_TITLE"}]),
    ("The Indoor Air Quality Specialist tested airflow rates in the hospital ward.",
     [{"text": "Indoor Air Quality Specialist", "label": "JOB_TITLE"}]),

    # --- Snowflake (ice crystal) ---
    ("The Meteorologist explained how each snowflake forms a unique crystal structure.",
     [{"text": "Meteorologist", "label": "JOB_TITLE"}]),
    ("A Children's Book Illustrator drew a magnified snowflake for the winter edition.",
     [{"text": "Children's Book Illustrator", "label": "JOB_TITLE"}]),
    ("The Photographer captured a single snowflake on a black velvet surface.",
     [{"text": "Photographer", "label": "JOB_TITLE"}]),

    # --- Docker (port worker) ---
    ("The Harbor Master managed a team of docker workers at the busy port.",
     [{"text": "Harbor Master", "label": "JOB_TITLE"}]),
    ("A Logistics Coordinator hired additional docker hands for the peak shipping season.",
     [{"text": "Logistics Coordinator", "label": "JOB_TITLE"}]),

    # --- Node (biology / graph theory) ---
    ("The Oncologist examined the lymph node for signs of cancer spread.",
     [{"text": "Oncologist", "label": "JOB_TITLE"}]),
    ("A Mathematician studied each node in the graph to find the shortest path.",
     [{"text": "Mathematician", "label": "JOB_TITLE"}]),
    ("The Surgeon removed a swollen lymph node during the biopsy procedure.",
     [{"text": "Surgeon", "label": "JOB_TITLE"}]),

    # --- Pandas (animal) ---
    ("The Zoologist studied giant pandas in the Sichuan province of China.",
     [{"text": "Zoologist", "label": "JOB_TITLE"}]),
    ("A Conservation Biologist worked to protect the habitat of wild pandas.",
     [{"text": "Conservation Biologist", "label": "JOB_TITLE"}]),
    ("The Wildlife Veterinarian treated a sick pandas cub at the breeding center.",
     [{"text": "Wildlife Veterinarian", "label": "JOB_TITLE"}]),

    # --- Git (dialect / British slang) ---
    ("The Linguist documented the use of the word git in British regional dialects.",
     [{"text": "Linguist", "label": "JOB_TITLE"}]),
    ("A Sociolinguist studied how the word git evolved in colloquial British English.",
     [{"text": "Sociolinguist", "label": "JOB_TITLE"}]),

    # --- Lambda (physics / Greek letter) ---
    ("The Physics Professor used lambda to represent the wavelength in the equation.",
     [{"text": "Physics Professor", "label": "JOB_TITLE"}]),
    ("A Fraternity Advisor organized the Lambda chapter's annual fundraising event.",
     [{"text": "Fraternity Advisor", "label": "JOB_TITLE"}]),
    ("The Nuclear Physicist calculated the lambda decay constant for the isotope.",
     [{"text": "Nuclear Physicist", "label": "JOB_TITLE"}]),

    # --- Mixed: real skill + ambiguous non-skill word ---
    ("The Backend Developer brewed Java coffee while debugging a PostgreSQL query.",
     [{"text": "Backend Developer", "label": "JOB_TITLE"},
      {"text": "PostgreSQL", "label": "DATABASE"}]),
    ("The ML Engineer watched a Monty Python marathon after deploying a TensorFlow model.",
     [{"text": "ML Engineer", "label": "JOB_TITLE"},
      {"text": "TensorFlow", "label": "PYTHON_ECOSYSTEM"}]),
    ("The SRE cleaned rust off the rack and then configured Docker containers.",
     [{"text": "SRE", "label": "JOB_TITLE"},
      {"text": "Docker", "label": "DEVOPS"}]),
    ("The Full Stack Developer placed a ruby ring on the table and opened a Redis terminal.",
     [{"text": "Full Stack Developer", "label": "JOB_TITLE"},
      {"text": "Redis", "label": "DATABASE"}]),
    ("The Frontend Developer appreciated angular architecture and built sites with Vue.js.",
     [{"text": "Frontend Developer", "label": "JOB_TITLE"},
      {"text": "Vue.js", "label": "JS_ECOSYSTEM"}]),
    ("The Data Scientist stored a flask in the lab then wrote a FastAPI endpoint.",
     [{"text": "Data Scientist", "label": "JOB_TITLE"},
      {"text": "FastAPI", "label": "PYTHON_ECOSYSTEM"}]),
]

# --------------------------------------------------
# Soft Negative Templates
# Generic role-based sentences with no technical skills.
# --------------------------------------------------

SOFT_NEGATIVE_TEMPLATES = [
    ("The Project Manager led the team through a challenging quarter.",
     [{"text": "Project Manager", "label": "JOB_TITLE"}]),
    ("An experienced Team Lead should communicate clearly with all stakeholders.",
     [{"text": "Team Lead", "label": "JOB_TITLE"}]),
    ("The Product Owner prioritized features based on customer feedback.",
     [{"text": "Product Owner", "label": "JOB_TITLE"}]),
    ("A Scrum Master facilitated the daily standup and removed blockers for the team.",
     [{"text": "Scrum Master", "label": "JOB_TITLE"}]),
    ("The Business Analyst gathered requirements from stakeholders across three departments.",
     [{"text": "Business Analyst", "label": "JOB_TITLE"}]),
    ("A Technical Writer documented the onboarding process for new hires.",
     [{"text": "Technical Writer", "label": "JOB_TITLE"}]),
    ("The QA Lead reviewed test results and reported the findings to management.",
     [{"text": "QA Lead", "label": "JOB_TITLE"}]),
    ("A UX Designer conducted user interviews to understand pain points in the workflow.",
     [{"text": "UX Designer", "label": "JOB_TITLE"}]),
    ("The Engineering Manager set quarterly goals and conducted performance reviews.",
     [{"text": "Engineering Manager", "label": "JOB_TITLE"}]),
    ("A Solutions Architect presented the proposed system design to the client.",
     [{"text": "Solutions Architect", "label": "JOB_TITLE"}]),
    ("The CTO addressed the board about the company's long-term technology vision.",
     [{"text": "CTO", "label": "JOB_TITLE"}]),
    ("A Release Manager coordinated deployments across multiple teams every two weeks.",
     [{"text": "Release Manager", "label": "JOB_TITLE"}]),
    ("The Support Engineer resolved customer tickets within the agreed service level.",
     [{"text": "Support Engineer", "label": "JOB_TITLE"}]),
    ("A Security Analyst reviewed access logs and flagged unusual activity.",
     [{"text": "Security Analyst", "label": "JOB_TITLE"}]),
    ("The Database Administrator performed routine maintenance during the weekend window.",
     [{"text": "Database Administrator", "label": "JOB_TITLE"}]),
    ("A Systems Administrator configured user accounts and managed permissions.",
     [{"text": "Systems Administrator", "label": "JOB_TITLE"}]),
    ("The IT Director approved the budget for the infrastructure upgrade.",
     [{"text": "IT Director", "label": "JOB_TITLE"}]),
    ("A Network Engineer troubleshot connectivity issues in the branch offices.",
     [{"text": "Network Engineer", "label": "JOB_TITLE"}]),
    ("The VP of Engineering hired three new team leads to support the growing organization.",
     [{"text": "VP of Engineering", "label": "JOB_TITLE"}]),
    ("A Technical Recruiter screened candidates and scheduled interviews with hiring managers.",
     [{"text": "Technical Recruiter", "label": "JOB_TITLE"}]),
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
# Hard / Soft Negative Generation
# --------------------------------------------------

def generate_hard_negative():
    """
    Picks a random hard or soft negative template, tokenizes it,
    and locates annotated entity spans via sliding-window matching.
    Returns {"tokenized_text": [...], "ner": [[start, end, label], ...]}.
    """
    pool = HARD_NEGATIVE_TEMPLATES + SOFT_NEGATIVE_TEMPLATES
    text, entities_meta = random.choice(pool)

    tokens = text.split()
    ner = []

    for ent in entities_meta:
        ent_tokens = ent["text"].split()
        ent_len = len(ent_tokens)

        for i in range(len(tokens) - ent_len + 1):
            sub = " ".join(t.strip(",.;") for t in tokens[i:i + ent_len])
            if sub.lower() == ent["text"].lower():
                ner.append([i, i + ent_len - 1, ent["label"]])
                break  # first match per entity is enough

    return {"tokenized_text": tokens, "ner": ner}


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
            # Hard / soft negative: ambiguous words without skill annotations
            dataset.append(generate_hard_negative())
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
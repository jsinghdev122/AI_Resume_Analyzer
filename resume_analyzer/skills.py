"""Skill extraction: a curated taxonomy matched with word-boundary-safe patterns.

Deliberately rule-based: every detected skill can be explained ("the text says X"),
and it needs no model or API call. Ambiguous words (Go, R, C, Excel, Spark) are only
matched through unambiguous aliases such as "golang" or "apache spark".
"""
import re
from typing import Dict, List, Tuple

from .preprocess import jd_lines

TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Programming languages": {
        "Python": ["python", "python3"],
        "Java": ["java"],
        "JavaScript": ["javascript", "ecmascript", "es6"],
        "TypeScript": ["typescript"],
        "C++": ["c++"],
        "C#": ["c#"],
        "C": ["c programming", "c language", "embedded c"],
        "Go": ["golang", "go programming"],
        "Rust": ["rust"],
        "Kotlin": ["kotlin"],
        "Swift": ["swift", "swiftui"],
        "PHP": ["php"],
        "Ruby": ["ruby", "ruby on rails", "rails"],
        "Scala": ["scala"],
        "R": ["r programming", "r language", "rstudio"],
        "MATLAB": ["matlab"],
        "Bash": ["bash", "shell scripting", "shell scripts"],
        "SQL": ["sql", "t-sql", "tsql", "pl/sql", "plsql"],
    },
    "Web & backend": {
        "FastAPI": ["fastapi"],
        "Flask": ["flask"],
        "Django": ["django"],
        "Node.js": ["node.js", "nodejs"],
        "Express": ["express.js", "expressjs"],
        "Spring Boot": ["spring boot", "springboot", "spring framework"],
        ".NET": [".net", "dotnet", ".net core", "asp.net"],
        "React": ["react", "react.js", "reactjs"],
        "Angular": ["angular", "angularjs"],
        "Vue": ["vue", "vue.js", "vuejs"],
        "Next.js": ["next.js", "nextjs"],
        "HTML": ["html", "html5"],
        "CSS": ["css", "css3", "sass", "scss"],
        "Tailwind CSS": ["tailwind", "tailwindcss"],
        "REST APIs": ["rest api", "rest apis", "restful", "restful apis", "restful services", "rest services"],
        "GraphQL": ["graphql"],
        "gRPC": ["grpc"],
        "Microservices": ["microservices", "microservice architecture"],
        "WebSockets": ["websockets", "websocket"],
        "Authentication": ["oauth", "oauth2", "jwt", "sso", "openid connect"],
    },
    "Data & machine learning": {
        "Machine Learning": ["machine learning", "ml models", "predictive modeling", "predictive modelling"],
        "Deep Learning": ["deep learning", "neural networks"],
        "NLP": ["nlp", "natural language processing", "text mining"],
        "Computer Vision": ["computer vision", "image processing", "object detection"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "TensorFlow": ["tensorflow"],
        "PyTorch": ["pytorch"],
        "Keras": ["keras"],
        "XGBoost": ["xgboost", "lightgbm", "gradient boosting"],
        "Pandas": ["pandas"],
        "NumPy": ["numpy"],
        "Matplotlib": ["matplotlib", "seaborn", "plotly"],
        "Apache Spark": ["apache spark", "pyspark"],
        "Hadoop": ["hadoop", "hdfs", "mapreduce"],
        "Apache Kafka": ["kafka", "apache kafka"],
        "Apache Airflow": ["airflow", "apache airflow"],
        "dbt": ["dbt"],
        "ETL": ["etl", "elt", "data pipelines", "data pipeline"],
        "Data Warehousing": ["data warehouse", "data warehousing", "data modeling", "data modelling"],
        "Statistics": ["statistics", "statistical analysis", "hypothesis testing"],
        "A/B Testing": ["a/b testing", "ab testing", "experimentation"],
        "Feature Engineering": ["feature engineering"],
        "MLOps": ["mlops", "mlflow", "model deployment", "model serving", "kubeflow"],
        "Time Series": ["time series", "forecasting"],
    },
    "GenAI & search": {
        "LLMs": ["llm", "llms", "large language models", "large language model"],
        "Generative AI": ["generative ai", "genai", "gen ai"],
        "Prompt Engineering": ["prompt engineering", "prompt design"],
        "RAG": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
        "LangChain": ["langchain", "llamaindex", "llama index"],
        "Hugging Face": ["hugging face", "huggingface", "transformers library"],
        "OpenAI API": ["openai api", "openai", "gpt-4", "chatgpt api"],
        "Vector Databases": [
            "vector database", "vector databases", "vector db", "vector search", "qdrant", "pinecone", "faiss",
            "chroma", "chromadb", "weaviate", "milvus", "pgvector",
        ],
        "Embeddings": ["embeddings", "embedding models", "sentence transformers", "sentence-transformers"],
        "Elasticsearch": ["elasticsearch", "opensearch", "solr"],
    },
    "Databases": {
        "PostgreSQL": ["postgresql", "postgres"],
        "MySQL": ["mysql", "mariadb"],
        "MongoDB": ["mongodb", "mongo"],
        "Redis": ["redis"],
        "SQLite": ["sqlite"],
        "Oracle Database": ["oracle database", "oracle db"],
        "SQL Server": ["sql server", "mssql", "ms sql"],
        "DynamoDB": ["dynamodb"],
        "Cassandra": ["cassandra"],
        "Snowflake": ["snowflake"],
        "BigQuery": ["bigquery"],
        "Redshift": ["redshift"],
        "NoSQL": ["nosql"],
    },
    "Cloud & DevOps": {
        "AWS": ["aws", "amazon web services"],
        "Azure": ["azure"],
        "Google Cloud": ["gcp", "google cloud"],
        "Docker": ["docker", "dockerfile", "docker compose", "containerization", "containerisation", "containers"],
        "Kubernetes": ["kubernetes", "k8s", "helm"],
        "Terraform": ["terraform"],
        "Ansible": ["ansible"],
        "CI/CD": ["ci/cd", "cicd", "ci cd", "continuous integration", "continuous delivery", "continuous deployment"],
        "Jenkins": ["jenkins"],
        "GitHub Actions": ["github actions"],
        "GitLab CI": ["gitlab ci", "gitlab-ci"],
        "Linux": ["linux", "unix", "ubuntu"],
        "AWS Lambda": ["aws lambda", "lambda functions", "serverless"],
        "Amazon S3": ["amazon s3", "s3"],
        "EC2": ["ec2"],
        "SageMaker": ["sagemaker"],
        "Nginx": ["nginx"],
        "Monitoring": ["prometheus", "grafana", "datadog", "observability", "cloudwatch"],
        "Infrastructure as Code": ["infrastructure as code", "iac", "cloudformation"],
    },
    "Testing & security": {
        "Pytest": ["pytest"],
        "Unit Testing": ["unit testing", "unit tests", "junit", "test-driven development", "tdd"],
        "Test Automation": ["test automation", "selenium", "cypress", "playwright", "jest"],
        "Security": ["owasp", "penetration testing", "application security", "encryption", "cybersecurity"],
    },
    "Mobile": {
        "Android": ["android"],
        "iOS": ["ios"],
        "React Native": ["react native"],
        "Flutter": ["flutter", "dart"],
    },
    "Tools & analytics": {
        "Git": ["git", "github", "gitlab", "bitbucket", "version control"],
        "Jira": ["jira", "confluence"],
        "Tableau": ["tableau"],
        "Power BI": ["power bi", "powerbi"],
        "Excel": ["microsoft excel", "ms excel", "advanced excel", "excel formulas", "pivot tables"],
        "Looker": ["looker", "looker studio", "data studio"],
        "Jupyter": ["jupyter", "jupyter notebook", "jupyter notebooks"],
        "Postman": ["postman", "swagger", "openapi"],
        "Figma": ["figma"],
    },
    "Engineering practices": {
        "Data Structures & Algorithms": ["data structures", "algorithms", "data structures and algorithms"],
        "Object-Oriented Programming": ["object-oriented programming", "object oriented programming", "oop", "oops"],
        "System Design": ["system design", "software architecture", "scalable systems", "distributed systems"],
        "Design Patterns": ["design patterns"],
        "API Design": ["api design", "api development", "api integration"],
        "Code Review": ["code review", "code reviews", "peer review"],
        "Agile": ["agile", "scrum", "kanban", "sprint planning"],
    },
    "Soft skills": {
        "Communication": ["communication skills", "communication", "communicating"],
        "Leadership": ["leadership", "team lead", "led a team", "leading a team"],
        "Mentoring": ["mentoring", "mentorship", "mentored"],
        "Problem Solving": ["problem solving", "problem-solving", "analytical skills", "analytical thinking"],
        "Collaboration": ["collaboration", "cross-functional", "cross functional", "teamwork", "stakeholder management"],
        "Project Management": ["project management", "project planning", "program management"],
    },
}


def _alias_pattern(name: str) -> str:
    parts = [re.escape(p) for p in re.split(r"[\s-]+", name.lower()) if p]
    return r"[\s-]+".join(parts)


# Names that are also ordinary words: only their explicit aliases may match ("golang", not "go").
_AMBIGUOUS_NAMES = {"Go", "R", "C", "Excel"}


def _compile() -> List[Tuple[str, str, "re.Pattern"]]:
    compiled = []
    for category, skills in TAXONOMY.items():
        for canonical, aliases in skills.items():
            names = {a.lower() for a in aliases}
            if canonical not in _AMBIGUOUS_NAMES:
                names.add(canonical.lower())
            alternatives = "|".join(sorted((_alias_pattern(n) for n in names), key=len, reverse=True))
            pattern = re.compile(rf"(?<![a-z0-9+#])(?:{alternatives})(?![a-z0-9+#])")
            compiled.append((canonical, category, pattern))
    return compiled


_PATTERNS = _compile()


def find_skills(text: str) -> Dict[str, str]:
    """Skills mentioned in the text, as {skill name: category}."""
    lowered = text.lower()
    return {name: category for name, category, pattern in _PATTERNS if pattern.search(lowered)}


_PREFERRED_RE = re.compile(
    r"nice[- ]to[- ]have|preferred|bonus|a plus|is a plus|are a plus|desirable|good to have|optional|"
    r"not required|advantage|added advantage",
    re.I,
)
_REQUIRED_HEADING_RE = re.compile(
    r"require|qualif|must|responsib|what you|looking for|you have|about you|the role|your (?:skills|experience)",
    re.I,
)


def jd_skill_importance(jd_text: str) -> Dict[str, Tuple[str, str]]:
    """Skills in a job description as {skill: (category, "required" | "preferred")}.

    A skill is "preferred" when its line says so ("nice to have", "a plus") or when it sits
    under a heading such as "Preferred qualifications". Everything else counts as required.
    """
    mode = "required"
    found: Dict[str, Tuple[str, str]] = {}
    for line, is_bullet in jd_lines(jd_text):
        is_heading = not is_bullet and len(line.split()) <= 6 and not line.endswith(".")
        if is_heading:
            if _PREFERRED_RE.search(line):
                mode = "preferred"
            elif _REQUIRED_HEADING_RE.search(line):
                mode = "required"
            continue
        line_mode = "preferred" if _PREFERRED_RE.search(line) else mode
        for skill, category in find_skills(line).items():
            previous = found.get(skill)
            if previous is None or (previous[1] == "preferred" and line_mode == "required"):
                found[skill] = (category, line_mode)
    return found

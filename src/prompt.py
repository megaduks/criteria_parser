from typing import Dict, List
from wasabi import msg
from dotenv import load_dotenv
import os

from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0,
)


def few_shot_entity_recognition(
    examples: List[Dict], criterion: str, entity: str
) -> List[str]:
    """Returns the predicted entities for a given criterion

    Args:
        examples (List[Dict]): List of few-shot examples
        criterion (str): Eligibility criterion
        entity (str): Entity type
    Returns:
        List[str]: List of predicted entities
    """
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()

    example_prompt = PromptTemplate(
        input_variables=["criterion", "entities"],
        template="criterion: {criterion} \n entities: {entities}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=f"Find examples of {entity} in the following criterion. {format_instructions}",
        suffix="criterion: {criterion} \n entities:",
        input_variables=["criterion"],
    )

    chain = few_shot_prompt | openai | output_parser

    msg.info(f"Sending query to {openai.model_name} model using few-shot learning.")
    return chain.invoke({"criterion": criterion})


if __name__ == "__main__":
    examples = [
        {
            "criterion": "The patient has a history of heart disease",
            "entities": "heart disease, heart attack",
        },
        {
            "criterion": "The patient has a history of diabetes",
            "entities": "diabetes, type 2 diabetes",
        },
        {
            "criterion": "The patient has a history of cancer",
            "entities": "cancer, lung cancer",
        },
        {
            "criterion": "The patient has a history of high blood pressure",
            "entities": "high blood pressure, hypertension",
        },
    ]

    criterion = "The patient has a history of heart attacks"
    entity = "disease"

    print(few_shot_entity_recognition(examples, criterion, entity))

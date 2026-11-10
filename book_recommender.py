import os
from typing import List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document


class BookRecommendation(BaseModel):
    title: str = Field(description="The title of the recommended book")
    author: str = Field(description="The author of the book")
    reason: str = Field(description="Why this book fits the user's mood and genre")

class BookRecommendationList(BaseModel):
    books: List[BookRecommendation] = Field(description="A list of three book recommendations")


def create_vector_store():
    """Creates a local in-memory vector database with a curated list of books."""
    book_database = [
        Document(page_content="Dune by Frank Herbert: A sprawling sci-fi epic about politics, religion, and survival on a harsh desert planet. Deeply thoughtful and philosophical.", metadata={"genre": "Sci-Fi"}),
        Document(page_content="Hyperion by Dan Simmons: A mind-bending space opera featuring seven pilgrims telling their harrowing tales on a world awaiting a mysterious creature called the Shrike. Very philosophical.", metadata={"genre": "Sci-Fi"}),
        Document(page_content="The Martian by Andy Weir: A gripping survival story filled with humor, science, and optimism about an astronaut stranded on Mars.", metadata={"genre": "Sci-Fi"}),
        Document(page_content="Project Hail Mary by Andy Weir: A lone astronaut must save Earth from an extinction-level threat, featuring high science and unexpected friendship. Inspiring and hopeful.", metadata={"genre": "Sci-Fi"}),
        Document(page_content="The Hobbit by J.R.R. Tolkien: A whimsical, cozy, yet adventurous fantasy journey across middle earth with Bilbo Baggins.", metadata={"genre": "Fantasy"}),
        Document(page_content="The Name of the Wind by Patrick Rothfuss: A lyrical, beautifully written fantasy masterpiece about a legendary wizard recounting his tragic youth.", metadata={"genre": "Fantasy"}),
    ]

    # Initialize Google's embedding model to convert text into numerical vectors
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
    
    # Store documents in a lightweight local vector store
    vector_store = InMemoryVectorStore.from_documents(book_database, embeddings)
    
    # Convert it into a retriever component
    return vector_store.as_retriever(search_kwargs={"k": 2})

def format_docs(docs):
    """Helper to join retrieved document contents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_recommendation(genre: str, mood: str):

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.7
    )

    retriever = create_vector_store()

    parser = PydanticOutputParser(pydantic_object=BookRecommendationList)
    # structured_llm = llm.with_structured_output(BookRecommendationList)

    prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful book recommendation assistant.\n"
                "Use ONLY the provided context pieces below to recommend books. Do not use outside knowledge.\n\n"
                "Context:\n{context}\n\n"
                "{format_instructions}"
            ),
            (
                "human",
                "Recommend books in the '{genre}' genre for someone who is feeling '{mood}'."
            )
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # chain = prompt | structured_llm
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(f"{x['genre']} {x['mood']}")),
            "genre": lambda x: x["genre"],
            "mood": lambda x: x["mood"]
        }
        | prompt
        | llm
        | parser
    )

    print(f"Finding books for a '{mood}' mood in the '{genre}' genre using Gemini...\n")

    result = chain.invoke({"genre": genre, "mood": mood})

    return result


if __name__ == "__main__":
    genre = "Fantasy"

    mood = "adventerous"

    try:
        recommendation = get_rag_recommendation(genre=genre,mood=mood)

        for i, book in enumerate(recommendation.books, 1):
            print(f"{i} {book.title} by {book.author}")
            print(f"    Why: {book.reason}\n")

    except Exception as e:
        print(f"An error occured {e}")
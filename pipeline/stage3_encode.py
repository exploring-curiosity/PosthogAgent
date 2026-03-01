"""
Stage 3: Encode — Convert behavioral descriptions into vector embeddings
using Mistral Embed.

With a single session, this proves the pipeline works.
With multiple sessions, embeddings can be clustered into persona archetypes.
"""

import json
from pathlib import Path
from mistralai import Mistral


def encode_behavior(behavioral_description: str, api_key: str) -> list[float]:
    """
    Encode a behavioral description into a vector embedding
    using Mistral Embed.
    """
    client = Mistral(api_key=api_key)

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[behavioral_description],
    )

    return response.data[0].embedding  # Returns a 1024-dimensional vector


def encode_multiple_sessions(session_descriptions: list[str], api_key: str) -> list[list[float]]:
    """
    Batch encode multiple session descriptions.
    Mistral Embed supports batch input.
    """
    client = Mistral(api_key=api_key)

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=session_descriptions,
    )

    return [item.embedding for item in response.data]


async def encode_and_save_async(
    behavioral_description: str,
    user_profile: dict,
    api_key: str,
    output_path: str | None = None,
) -> dict:
    """Async version of encode_and_save."""
    client = Mistral(api_key=api_key)

    response = await client.embeddings.create_async(
        model="mistral-embed",
        inputs=[behavioral_description],
    )

    embedding = response.data[0].embedding

    result = {
        "session_id": user_profile.get("session_id"),
        "demographics": user_profile,
        "embedding_dim": len(embedding),
        "embedding": embedding,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

    return result


def encode_and_save(
    behavioral_description: str,
    user_profile: dict,
    api_key: str,
    output_path: str | None = None,
) -> dict:
    """Run Stage 3 and optionally save the output."""
    embedding = encode_behavior(behavioral_description, api_key)

    result = {
        "session_id": user_profile.get("session_id"),
        "demographics": user_profile,
        "embedding_dim": len(embedding),
        "embedding": embedding,
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved {len(embedding)}-dim embedding to {output_path}")

    return result


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import MISTRAL_API_KEY, DESCRIPTIONS_DIR, EMBEDDINGS_DIR, ensure_data_dirs, validate_api_keys

    validate_api_keys()
    ensure_data_dirs()

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.stage3_encode <description.txt>")
        sys.exit(1)

    desc_path = sys.argv[1]
    with open(desc_path) as f:
        description = f.read()

    # Extract session_id from filename (description_<session_id>.txt)
    stem = Path(desc_path).stem
    session_id = stem.replace("description_", "")

    output_path = str(EMBEDDINGS_DIR / f"embedding_{session_id}.json")
    result = encode_and_save(
        description,
        {"session_id": session_id},
        MISTRAL_API_KEY,
        output_path,
    )
    print(f"Generated {result['embedding_dim']}-dimensional embedding")

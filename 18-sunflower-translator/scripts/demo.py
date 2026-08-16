import httpx

API = "http://localhost:8018"


def main():
    client = httpx.Client(base_url=API, timeout=120)

    print("Sunflower-14B Multilingual Demo")
    print("=" * 50)

    print("\n1. Supported languages:")
    r = client.get("/languages")
    langs = r.json()["supported_languages"]
    for code, name in list(langs.items())[:6]:
        print(f"   {code}: {name}")
    print(f"   ... and {len(langs) - 6} more")

    print("\n2. Translate English → Luganda:")
    r = client.post("/translate", json={
        "text": "Good morning! How are you today?",
        "source_language": "eng",
        "target_language": "lug",
    })
    data = r.json()
    print(f"   EN: {data['original']}")
    print(f"   LG: {data['translation']}")

    print("\n3. Chat in Acholi:")
    r = client.post("/chat", json={
        "message": "What is artificial intelligence?",
        "language": "ach",
    })
    print(f"   {r.json()['response'][:200]}")

    print("\n4. Detect language and translate:")
    r = client.post("/detect-and-translate", json={
        "text": "Oli otya?",
        "target_language": "eng",
    })
    data = r.json()
    print(f"   Detected: {data['detected_language']}")
    print(f"   Translation: {data['translation']}")


if __name__ == "__main__":
    main()

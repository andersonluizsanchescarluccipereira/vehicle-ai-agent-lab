"""Validação estrutural mínima do OKF v0.2."""
from src.knowledge import list_concepts

if __name__ == '__main__':
    concepts = list_concepts()
    print(f'OKF estrutural: {len(concepts)} documentos tipados válidos.')
    for concept in concepts:
        print(f" - {concept['id']} [{concept['type']}] ({concept['status']})")

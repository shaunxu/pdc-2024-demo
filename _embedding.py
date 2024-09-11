from transformers import AutoTokenizer, AutoModel
import torch

class Embedding:

    device: torch.device
    tokenizer: AutoTokenizer
    model: AutoModel

    def __init__(self, model_path: str) -> None:
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).eval().to(self.device)
    
    def embed(self, sentences: list[str], normalize_embeddings: bool = True) -> list[list[float]]:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        if normalize_embeddings:
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()
"""The citation classifier: an NLI cross-encoder, premise = evidence, hypothesis = citing sentence."""
MAX_LEN = 512


class Classifier:
    def __init__(self, name, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(name)
        # Published in fp16; fp32 on CPU and MPS, where fp16 is slow or unsupported.
        dtype = torch.float16 if str(device).startswith('cuda') else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, torch_dtype=dtype).to(device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        expected = {'ACCURATE', 'NOT_ACCURATE', 'NOT ENOUGH INFO'}
        if set(self.labels) != expected:
            raise ValueError(f'{name} has labels {self.labels}, not a CiteAudit classifier '
                             f'(expected {sorted(expected)})')

    def __call__(self, evidence, statement):
        """Label probabilities for one citation; evidence is a list of sentences, best first."""
        enc = self.tok(' '.join(evidence), statement, truncation='longest_first',
                       max_length=MAX_LEN, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            p = self.model(**enc).logits.float().softmax(-1)[0].tolist()
        return dict(zip(self.labels, p))

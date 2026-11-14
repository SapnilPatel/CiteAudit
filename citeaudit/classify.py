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
        # Per-class decision biases chosen on cross-validation (scripts/tune_bias.py), stored
        # with the model they were tuned for. Added to the logits, so the reported probabilities
        # agree with the label.
        bias = getattr(self.model.config, 'citeaudit_bias', None) or {}
        self.bias = torch.tensor([float(bias.get(c, 0.)) for c in self.labels], device=device)

    def __call__(self, evidence, statement):
        """Label probabilities for one citation; evidence is a list of sentences, best first."""
        enc = self.tok(' '.join(evidence), statement, truncation='longest_first',
                       max_length=MAX_LEN, return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            p = (self.model(**enc).logits.float()[0] + self.bias).softmax(-1).tolist()
        return dict(zip(self.labels, p))

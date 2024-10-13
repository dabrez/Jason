from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline


tokenizer = AutoTokenizer.from_pretrained("laiyer/distilroberta-bias-onnx")
model = ORTModelForSequenceClassification.from_pretrained("laiyer/distilroberta-bias-onnx")
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
)

classifier_output = classifier("Your text to analyze for bias.")
score = (classifier_output[0]["score"] if classifier_output[0]["label"] == "BIASED" else 1 - classifier_output[0]["score"])

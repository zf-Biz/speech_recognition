from transformers import MBartForConditionalGeneration, MBartTokenizer


def main():
    with open("transcribed_text.txt", "r", encoding='utf-8') as text_file:
        # Read the text from file
        text = text_file.read()

    # Summarize the text
    summary = summarize_text(text)

    # Save the summary to a new file
    with open("summary.txt", "w", encoding='utf-8') as file:
        file.write(summary)

    print("Original Text:", text)
    print("Summary:", summary)


def summarize_text(text):
    """
    Summarize the input text using mBART model.

    Args:
        text (str): Text to summarize.

    Returns:
        str: Summarized text.
    """
    # Load mBART multilingual model and tokenizer
    model_name = "facebook/mbart-large-cc25"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBartTokenizer.from_pretrained(model_name)

    # Tokenize the input text in Lithuanian
    model_inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the summary (output)
    summary_ids = model.generate(
        model_inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode the generated summary and return as text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    main()
